CDF       
      obs    M   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?��1&�y     4  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       Mţt   max       P��     4  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ��Q�   max       <���     4      effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�(�\   max       @FB�\(��       !H   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��(�\    max       @vf�Q�       -P   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @'         max       @Q�           �  9X   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�X        max       @��@         4  9�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �I�   max       <���     4  ;(   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�P�   max       B/�     4  <\   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A��G   max       B/�~     4  =�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >�le   max       C��     4  >�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >��
   max       C��     4  ?�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          q     4  A,   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          A     4  B`   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          3     4  C�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       Mţt   max       Pl�f     4  D�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?���ᰉ�   max       ?�64�K     4  E�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��Q�   max       <���     4  G0   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�(�\   max       @F9�����       Hd   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��(�\    max       @vf�Q�       Tl   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @'         max       @Q�           �  `t   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�X        max       @�'          4  a   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         >�   max         >�     4  bD   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?a4�J�   max       ?�4m��8�       cx   !   
            $            !      (                2      4               N            p               	                                 !         &            &                        ,            (             $            	            &       O��JN;�O -/N5��N��rP"5RNp�O�N�� O���O�^Oե�N�\P:u�N4�AO��PV�PN�"�P;�OOMWN&3�O�W�O�Pf��O�*3OPz~N4?�P��O�H�N�AKM�LN9�N��;OW;�N���N�YyN\��OG��N;�NG�(NW�,N	=O9�fN��N0�NR�fO���O"N��N�h�Pw�O���ODWNɆ�O���O8�)OZZ�N��Ov�Ng-gO"�NĨ�O�NO}�+MţtO�<�O��O��cO��O$-SO��O�_NƻO�u�O��~N羼N g�<���<���<ě�<�9X<�9X<49X���
�ě��o�o�#�
�#�
�D���e`B�e`B�e`B��o��o��C���C���t����㼣�
��1��j��j�ě����ͼ��ͼ�/��`B���+�C��C��C��C��t��t�����w�0 Ž49X�49X�49X�49X�49X�8Q�<j�<j�<j�<j�@��D���H�9�L�ͽP�`�T���Y��Y��Y��]/�e`B�m�h�u�}󶽁%��%�����C���C���O߽�O߽�O߽����1��Q���������������������	#)*) 								������������yz~��������zyyyyyyyy&)58;?>;5.)#"$&&&&&&_mz�����������zkaa__@BEOOYOB?9@@@@@@@@@@#/<DHQTRH</#
�������������������������.67)�������JN[gn�������tocNHDBJ��������������������

������������TXWa������������zaWT��������������������)5BNUONHB5)��������-$�������S[ht������xthd[XSSSSHaz�������vaTKC@=>@Hw������������������w�����������������������*6CL]cd\C���#/<GGF?</#|��������������{uuw|������
���������"-/<Uab]UH</$�����


���������HP\Z`gt���������tNBH/<HUY[YUVQMHF<#��������������������'),66:6))'''''''''''"./6/"BHUacinunaUHG@BBBBBB��)5;BLB5)����w�������������wwwwww	
 #$/;><0/##
	#%,,)#~���������������xwz~������������������������������������������������������������OOU[hiih][XOOOOOOOOO������������������������	����������������

������������NOV[fhjnh[ONNNNNNNNNlz{{�������������thl<BO[htxzzxth[VOBB;<<Zamsz|~���zzmfa\ZZZZ�����������������������&(&#��������)5BIRZ^b_WNB5)# !)ghirt���������tqhfdg./<GHRKHG></)*+,....)0:IUbnvyym[UI60##)\aenz��������znaa\Z\�����������������~}�����������������������
 #$#"
����������������������������������������������`gstx�����thggc`````t|��������������tllt��5BKGB?3)�������������������������	#0<IU[ZI;5220.
	BGN[gtz~��tg[NE@>@B����������������������������������������:<IUZbcgljb[UOI<504:x{����������{vtw{~xx)/<CHUaiaZUH@</+*)))��������������������  �������PUanx�����zvliaUKJLP�����������������������������������������������������
��#�8�H�I�S�M�<�/�#��
���ֺҺҺֺ���������ֺֺֺֺֺֺֺ��y¦²¿����������²¦����ùôìëìù������������������������ƎƊƅƎƚƧƳ������ƼƳƧƚƎƎƎƎƎƎ��������������(�6�A�Z�h�o�h�O�C�*���U�K�U�a�a�n�v�q�n�a�U�U�U�U�U�U�U�U�U�U������"�%�/�;�H�S�[�_�a�U�T�;�/�"��ּ˼ʼǼ��üƼʼʼּݼ�������ּּ����r�g�g�l�u�{����������������¼����������������������Ŀѿݿ��ݿѿĿ��������ѿĿ����������Ŀѿݿ������
����ݿѼY�M�M�C�M�Y�f�i�p�f�Y�Y�Y�Y�Y�Y�Y�Y�Y�YĲĦčĀĚĝĜĒĚĦĳ��������������Ĳ�h�`�[�Q�Q�[�h�j�k�t�v�t�h�h�h�h�h�h�h�h�����������������������������������������n�N�0����)�nŠ������������ŹŮŠŇ�n�/�,�(�#�(�/�<�H�U�V�U�T�K�H�=�<�/�/�/�/��
�����(�5�Z�g�����������g�N�5�(��Y�R�S�U�V�f�r�����������������f�`�[�Y�4�4�0�+�4�A�C�G�E�A�4�4�4�4�4�4�4�4�4�4�T�G�;�.���������	�"�.�;�J�I�M�Z�Y�[�T�$�������$�0�=�?�B�@�=�3�0�$�$�$�$�����g�V�T�V�`�m�y�����ѿ���#�����Ŀ������������������������������������������������s�g�Z�X�S�^�e�g�s������������������ ������������������������²�N�=�0�+�/�B¦¿��������
���s�f�N�B�<�=�A�M�Z�f�s�������������������������������������������������������ìëìøù��������ùìììììììììì�/�&�)�/�/�;�D�A�<�;�/�/�/�/�/�/�/�/�/�/ùòôöù������������������ùùùùùù��v�����������������������������������àÚÓÐÇÄÇÓàìîñìëààààààìëàÖÓÐÓÚàìîùû����������ùì�U�T�Q�U�a�n�zÄ�z�x�n�a�U�U�U�U�U�U�U�U�V�P�I�=�5�:�=�E�I�V�b�o�{�~ǆǄ�{�o�b�Vǈ�~�{�z�{ǅǈǉǔǠǘǔǈǈǈǈǈǈǈǈ������$�0�2�0�.�$�$�����������������������������������þ���������������	�����'�/�'�������������������������������%�(�#�!�����D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D����������������������������������������һ��������������û˻ϻŻû����������������e�Y�L�3�+�)�,�7�=�E�Y�r�~�����������s�e���������������������������������������������������������������������������������������������������������������ßÞàìñ���������+�.�,������ùìß�^�U�`�n�zÇÓàìù����úöìÓÇ�z�n�^���������y�������������½ĽʽƽĽ�������D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D����s�g�Z�S�O�Q�Z�g�s���������������������ʾ������������ʾ׾ܾ�����������׾ʾ����������(�4�?�A�D�B�;�4�-�(��ŠŝŔňōŔŠţŭŹſ��ŹŭŠŠŠŠŠŠEEEEE*E-E7ECENEPEYE\E`E]EUEPECE7E*E�:�.�9�:�F�S�_�l�m�l�c�_�S�F�:�:�:�:�:�:�x�m�w�x���������������Żû������������x��������������������������������������������������������	��'�.�4�1�"�	�����������	���"�.�3�:�?�?�;�.�"������~�r�m�r�~�~�~�����������������������Ľ������������������ɽݽ�������н�Ā�t�o�wăđĚĦĳĿ����������ĿĳĦčĀ����������ĽĿ������������ �%�����������������������������������������	��������������»ûлܻ����������ܻлû����������'�4�@�J�J�@�4�'������EuEpEsEuEyE~E�E�E�E�E�E�E�E�E�E�E�E�E�Eu�l�g�`�_�`�l�w�y�~�������y�m�l�l�l�l�l�l���������������̽���
�������нĽ����������ùܹ�����#�������Ϲù�������(�5�7�9�5�0�(������������������ʼμּ���߼ּʼ��������������� ; I O P P , a ( X P F = K 2 Z B | K L c ` k = F  P V Q B 9 g a V � Q ? G ) ( T p > , , V " > D x ^ = < ) H [   q A o _   i S [ ? : 6 [ ) G O Y P l  ^    =  `  P  k  �  �  5  �      $  �  *  ;  b  I  4    �  �  R  h  ?    E  �  Q  �  /      z  �  �  �    ~  �  R  �  �  0  �    G  b  @  q  �     �  M  B  �    �  �  �  @  �  �  �  �  N  &  d  I  �  U  ^  O  X  D  v  �  �  C��`B<�o<D��<�C�<���ě���o���ͼ��ͽ'ě��L�ͼ�o�8Q켃o�t������`B��C��49X��1�49X�\)�ȴ9�H�9�8Q��`B�I��T���C����\)�'@��49X�H�9�'Y��,1�'0 Ž@���7L����@��T�����T����T���Y���1��7L�y�#��7L��C���C����T�ixսƧ�m�h��hs�y�#�\�� Ž�%��vɽȴ9��^5���罡�����-��{��t���^5��S���h��v�B�B��B�pB WiBsB ��B�B:�B�TB�+B	F�BŶB$�B ��BI�B�HB`�BA]A�  B @�BPpB/�BeB*�UB6^B��B#�B
ӽB�vBBe�A�P�BݮBL�B
�	B��B��B
��B��B(hB!r\BK�B��B��B0�BR�B #�BL'A�HSBJB�}B��B��B��B&�kB�Bq�BDLB��B�B �B	��B��B�B/FB%ĜBߩB5�B�B&�,B)n�B=B9�B��B�fB�eB,cJB�xB�4B�ZB A<B,�B ZTB@ZB1*B��B�xB	3�B��B#� B /B�aBG�BǔBE�A�A�B ?�B=B/�~B��B*�
B9�B��B#��BC>B�[B>+B�IA��GB�[B?�B
��B�B¼B
��B�@BwWB!�JBD�B��B��B�BJ�B @�B;�A�w�BCRB=B?�B�[B��B&�NB߇B��B?�B��B1B?�B	��B�B�BBEB%BjB��B��BO?B&��B)AgB�OB>|B��B?�B�-B,~YA���@A�A�c�A���B��A�e�Aƛ0A�f-AQM@�%Aw�A}1?@�`#A��A�:�A�y�A�==AÔ�A���@�Q�A9��Ab�B	��Aw/�A��A��F@���A���AA�rA�>�Aʹ�A���AΦ�A�kA��A̳2A��B4�B��B	��AKfx@�wA��hC�&A��
@���?���A���B�FB�A��uA�9jA"6C��PA�ȝASX�A5��A��C���@��@��YA���A���A^��@W@A&��A�gMA�A+A�/@��g@ʢ�C��A\�A,��>�leA�p|@��jA�~�@C��A���A�uHB��A���Aƀ�A��A��@��Av�_A}��@���A�A���A�>�A���AâBA�l@�+A:poA^��B	��Ax!6A�{�A���@�F+A�u;AA �A�.8A�sPA�Z�A��A���A�}�A�uA�P�BBDB�B	��AK@�@���A�x�C�#�A�Tj@�Y�?�!�A��B�EB��A�}�Aˀ.A"�OC��XA��AS��A6��A�d�C���@���@��A��A��uA_@]SA%XA�qzA�y�A��@��@��C��A��A/�>��
A��@�ߥ   !   
            $            "      )                2      4               N            q               	                                 !      	   &            '                        -            (             $            
            '   !                     +            !      !      -         ?      -         %      7            A                                                         '            %            #                        #         !      !                  #                                                   -         #      !         !      -            3                                                         '            !            #                        !                                          O%s�N;�N�-N5��N��rO���Np�O �N�3ZO��,O�^O/��N�\P:u�N4�AOXRO���N|��O�N�g�N&3�O��O�P0AXOF��O#u�N4?�Pl�fOhMJN��M�LN9�N��;OW;�Nv�3Nh��N\��O%��N;�NG�(NW�,N	=O	::N�DN0�NR�fO���O"NJ_�N�h�O��O�ODWNɆ�O���O&�(OZZ�N��N�Ng-gO"�NĨ�O���Od1MţtO}�O��O��SO��N蔄N�PO�_NƻO�9O��~N羼N g�  [  q  �  $  k  �  t  c  F  n  _  �  �  c  "  �  �  �  S  K  �  R  �  >  ]  �  Y  
    �  6    F  �  �  V  �  �  R  �  �      
%  �  �  $  �    �  �  �    n  q  M  *  �  V  7  �  �  n  I  �    �  �    Z  �  �  \  �    �  �<e`B<���<�j<�9X<�9X%   ���
�49X�49X�T���#�
��/�D���e`B�e`B��t�������C����ͼ�t���9X���
�\)�������ě��Y���/��`B��`B���+�C��\)��w�C����t�����w�0 ŽD���8Q�49X�49X�49X�8Q�@��<j�P�`�D���@��D���H�9�P�`�P�`�T���e`B�Y��Y��]/�m�h�u�u��o��%��+�����\)��O߽�O߽�O߽�hs�����1��Q���������������������	#)*) 								�������������yz~��������zyyyyyyyy&)58;?>;5.)#"$&&&&&&ioz����������zpjjihi@BEOOYOB?9@@@@@@@@@@#/<<HKNKH<8/#������������������������)34)������JN[gn�������tocNHDBJ�����������������������

������������TXWa������������zaWT��������������������
)5BHJHEB<5)	
��������������������fht�����thhaffffffffHMYamz�����|rmaTMHFH������������������������������������������*7CIY]]\OC*��#/<GGF?</#|����������������{y|��������������������"%/1<FHU]ZURHC</#"�����


���������it�����������tg`fefi (/<HUWYXTTNH</#  ��������������������'),66:6))'''''''''''"./6/"BHUacinunaUHG@BBBBBB��)5;BLB5)����{�������������{{{{{{#+/3/,##%,,)#���������������{y{������������������������������������������������������������OOU[hiih][XOOOOOOOOO�����������������������������������������

������������NOV[fhjnh[ONNNNNNNNNlz{{�������������thl<BO[htxzzxth[VOBB;<<[amqzz{zmia][[[[[[[[������������������������"#"��������")5BGPW\`]UNB5)$  ""ghirt���������tqhfdg./<GHRKHG></)*+,....)0:IUbnvyym[UI60##)]afnz�������znba\Z]]�����������������~}������������������������
"
�����������������������������������������������`gstx�����thggc`````t�������������tmllt�)5CDB>51)�������������������������
#0<DIUWWI930#
BGN[gtz~��tg[NE@>@B����������������������������������������7<AIUV_abdb`UIE<9477y{������������}{wuyy)/<CHUaiaZUH@</+*)))�����������������������������PUanx�����zvliaUKJLP�����������������������������������������
������������
��#�/�8�<�?�<�6�/�#��
�ֺҺҺֺ���������ֺֺֺֺֺֺֺ��¦²¿����¿¾²¦����ùôìëìù������������������������ƎƊƅƎƚƧƳ������ƼƳƧƚƎƎƎƎƎƎ����������������6�F�R�T�N�C�6�*����U�K�U�a�a�n�v�q�n�a�U�U�U�U�U�U�U�U�U�U�/�%�"���"�"�,�/�;�H�K�S�T�X�Y�T�H�;�/�ּμʼʼżɼʼּڼ�������ּּּּ��r�k�j�o�x�~�������������������������������������������Ŀѿݿ��ݿѿĿ��������ݿԿѿȿĿ������Ŀѿݿ����� � ������ݼY�M�M�C�M�Y�f�i�p�f�Y�Y�Y�Y�Y�Y�Y�Y�Y�YĲĦčĀĚĝĜĒĚĦĳ��������������Ĳ�h�`�[�Q�Q�[�h�j�k�t�v�t�h�h�h�h�h�h�h�h�����������������������������������������b�U�I�>�7�C�U�b�{ōŠŬŮŨŠŘŁ�{�n�b�/�/�'�.�/�<�H�R�Q�H�H�<�/�/�/�/�/�/�/�/�5�(������(�5�A�Z�g�����������s�N�5�r�f�f�^�d�f�h�r���������������������r�4�4�0�+�4�A�C�G�E�A�4�4�4�4�4�4�4�4�4�4�G�;��	� ����	�"�.�;�>�F�E�H�J�N�S�K�G�$�������$�0�=�?�B�@�=�3�0�$�$�$�$�Ŀ����p�f�a�a�k�y�����Ŀ���������������������������������������������������s�g�[�Z�U�Z�\�b�g�s�{�����������������s�� ���������������������[�F�?�?�F�N�Y²������� ����²¦�t�[�f�Z�P�C�=�?�A�M�Z�f�s��������������s�f����������������������������������������ìëìøù��������ùìììììììììì�/�&�)�/�/�;�D�A�<�;�/�/�/�/�/�/�/�/�/�/ùòôöù������������������ùùùùùù��v�����������������������������������àÝÓÒÇÅÇÓàìíðìæàààààààÞÛàåìðù����ùìàààààààà�U�T�Q�U�a�n�zÄ�z�x�n�a�U�U�U�U�U�U�U�U�V�S�I�=�8�>�I�J�V�b�o�y�{�ǃǁ�{�o�b�Vǈ�~�{�z�{ǅǈǉǔǠǘǔǈǈǈǈǈǈǈǈ������$�0�2�0�.�$�$�����������������������������������þ���������������	�����'�/�'����������������������������!�%� �����D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D����������������������������������������һ��������������û˻ϻŻû����������������e�Y�L�3�+�)�,�7�=�E�Y�r�~�����������s�e���������������������������������������������������������������������������������������������������������������ìæäçù�����������'�$�������ùì�n�b�Y�b�n�zÇÓàìùÿ��öíàÓÇ�z�n���������y�������������½ĽʽƽĽ�������D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D����s�g�Z�S�O�Q�Z�g�s���������������������ʾ¾����������ʾ׾����������׾ʾʾ����������(�4�?�A�D�B�;�4�-�(��ŠŝŔňōŔŠţŭŹſ��ŹŭŠŠŠŠŠŠEEEE!E*E/E7ECEPE\E^E\E\ESEPECE7E*EE�:�.�9�:�F�S�_�l�m�l�c�_�S�F�:�:�:�:�:�:�x�m�w�x���������������Żû������������x������������������������������������������������������	��&�-�3�0�"��	��� �����	���"�+�.�2�9�>�<�;�.�"������~�r�m�r�~�~�~�����������������������Ľ����������������������Žݽ������н�Ā�t�o�wăđĚĦĳĿ����������ĿĳĦčĀ������������������������� �������������������������������������������	����û����������ûлֻܻ�������ܻлûü���	����%�'�4�6�@�H�H�@�>�4�'��EuEpEsEuEyE~E�E�E�E�E�E�E�E�E�E�E�E�E�Eu�l�g�`�_�`�l�w�y�~�������y�m�l�l�l�l�l�l���������ѽ�������������ݽнĽ����������ùܹ�����#�������Ϲù�������(�5�7�9�5�0�(������������������ʼμּ���߼ּʼ��������������� 2 I J P P & a % G E F 8 K 2 Z 8 ? . O V ` _ = @  > V R ? > g a V � K $ G ( ( T p > . ) V " > D P ^ = A ) H [   q @ o _   c P [ 7 : 0 [ '  O Y ; l  ^    n  `    k  �  �  5  Y  �  F  $  |  *  ;  b  �  �  �  0  �  R  �  ?  8  �  h  Q  !  �  �    z  �  �  �  o  ~  b  R  �  �  0  =  �  G  b  @  q  u          B  �    b  �  �  $  �  �  �  �    &    I  M  U  �  �  X  D    �  �  C  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  �  �    /  D  S  Z  X  L  7    �  �  /  �  k    �  .  �  q  [  F  /    �  �  �  �  p  E    �  �  q  1  �  �  1   �  �  �  �  �  �  �  |  ]  :    �  �  �  D    �  r  �  �  �  $  
  �  �  �  �  �  l  O  /    �  �  �  �  �  {  N  !  �  k  e  `  [  U  P  K  E  ?  9  3  -  '       �   �   �   �   �    G  n  �  �  �  �  �  �  y  `  E  &  �  �  p    �  A  7  t  ~  �  t  h  \  O  F  d  l  N  .    �  �  �  v  M  "  �  "  <  L  W  `  c  `  Y  M  A  1    �  �  �  �  �  {  q  9  �  .  @  E  =  )  
  �  �  r  1  �  �  R    �  �  |  �  ;    E  [  m  a  P  8      �  �  �  M    �  �  I  �  �  �  _  W  O  E  :  )      �  �  �  �  �  �  z  V  1    �  �  j  �    B  p  �  �  �  �  �  �  �  s  B  	  �  k  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  c  T  D  >  4  ,  &      �  �  �  }  B    �  �  _  !  �  "          �  �  �  �  �  �  �  �  �  �  �  {  k  \  M  �  �  �  �  �  �  �  �  �  �  }  [  0     �  �  ;    �  �  b  }  e  9    }  �  h  f  X  ?     �  �  �  x    �  4  �  v  w  y  ~  �  �    w  l  \  I  5      �  �  �  S    �  q  �  �    0  F  Q  Q  ;    �  �  p  "  �  U  �  �  �  p  �         7  I  H  D  ;  %  	  �  �  �  _  "  �    5  [  �  �  �  z  p  e  [  R  I  @  7  .  %            �  �  7  :  L  I  9  '        �  �  �  �  X    �  �  e    �  �  �  w  k  b  T  ?  '    �  �  �  M    �  �  �  o  Z  V  �    2  >  ;  ,    �  �  �  U  
  �  ;  �  3  �  �  �  A  �    8  H  U  ]  Z  J  3    �  �  �  v  I    �  �  �  M  �  �  �  �  �  �  �  �  �  d  :    �  �  |  R  5    �  o  Y  J  <  -        �  �  �  �  �  �  �  �  i  R  :  "  
  �  	i  	�  	�  
  
  	�  	�  	�  	�  	D  �  r    T  �  �    �                �  �  �  z  F    �  �  r  !  �    g   �  �  �  �  �  �  �  �  �  �  �  �  }  s  h  ]  O  B  4  &    6  -  $        �  �  �  �  �  �  �  z  ^  A  $    �  �      �  �  �  �  �  �  z  e  Q  =  *      �  �  �  �  �  F  7  )      �  �  �  �  �  �  �  ~  W  0    �  �  �  �  �  �  �  �  �  �  �  �  �  Z  2    �  �  �  g  +    �  �  �  �  �  �  �  x  b  G  +    �  �  �  �  �  p  c  Z  P  G    #  ,  7  D  N  T  T  N  @  &    �  �  +  L  Q  N  I  A  �  �  �  �  �  �  �  �  �  �  �  �  ~  u  l  a  V  D  .    �  �  �  �  �  �  �  �  ~  f  D    �  �  �  Z    �  �  Z  R  G  =  3  %      �  �  �  �  �  �  p  V  ;  !    �  �  �  �  �  �  �  �  �  �  ~  s  i  ^  T  H  7  '      �  �  �  �  �  �  �    w  n  e  \  Q  D  6  (    
   �   �   �   �        �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  v  i  
              �  �  �  �  n  ;     �  �  r  6  �  v  
	  
"  
  
  	�  	�  	�  	i  	5  �  �  b      �  L  �    g  X  �  |  w  r  n  i  d  ]  V  N  F  ?  7  2  2  2  3  3  3  3  �  �  �  �  �  �  m  X  @  (    �  �  �  �  �  �  �  �  �  $        �  �  �  �  I    �  ~  `  ,  �  �  R  .  �  ~  �  �  �  �  �  �    d  F  %    �  �  F  �  �  H  �  :   �            �  �  �  �  �  �  e  D    �  �  F     �   �  �  �  �  �  �  �  �  �  {  a  G  +  
  �  �  �  o  I  %    �  �  �  �  �  �  �  u  S  %  �  �  k  &  �  �    �  �   �  �  �  �  �  �  �  �  �  ^  1  �  �  v  $  �  l    �  �  -    ~  {  v  l  a  U  G  6  %      �  �  �  �  o  M  �  N  n  Y  >  /    �  �  �  �  �  U    �  n    �  {  3  �  �  q  k  X  B  ,    �  �  �  �  �  �  y  _  @    �  �  h  \  E  M  @  1    
  �  �  �  �  �  v  [  A  &  
  �  �  �  �  *    �  �  �  �  h  B    �  �  �  @  �  r  �  c  �    L  �  �  �  {  p  f  \  R  I  @  7  /  $    
  �  �  �  �  �  ;  U  P  6    �  �  x  8  
�  
�  
E  	�  	<  y  �  �  �  C  �  7  /  '          �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  b  9    �  �  p  6  �  �  �  �  w  7  �  �  �  �  �  �  x  k  ]  N  >  /  '          �  �  �  .  m  Z  <    �  �  V  )    �  �  x  C  �  �  a  �  �  	  ?  G  B  3    �  �  �  y  5  �  �  V    �  h  '  �  �  q  �  �  �  �  �  �  �  �  q  \  G  2       �   �   �   �   �   {  �          �  �  �  �  �  c  <    �  �  p    �  '    �  �  h  B    �  �  �  G    �  o  #  �  �  1  �  j  �  {  �  �  �  �  �  �  �  �  �  �  u  U  2  	  �  �  M  �  �  �    �  �  �  �  �  \  /  �  �  �  �  �  �  �  _  3    �  �  R  V  Y  Z  W  R  L  F  @  ;  5  .  '         �  �  �  �  @  f  �  �  x  j  [  K  9  &    �  �  �    I     �   t     �  |  N    �  �  �  J    �  u  %  �  v    �    �     �  \  V  P  J  C  =  7  3  0  ,  )  &  #  $  0  ;  F  Q  ]  h  �  �  �  �  �  �  m  F    �  �  �  U    �  �  ~  X  �   �    �  i  |  �  �  �  �  R    �  �  a    �  W  �  �  �  �  �  {  T  (  �  �  �  y  F    �  �  I  �  �  )  �  1  �    �  �  �  �  �  �  �  |  n  `  R  D  6  )      
  �  �  �