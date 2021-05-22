CDF       
      obs    R   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?�|�hr�     H  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M��B   max       P�9)     H  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ���   max       <��     H   <   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?G�z�H   max       @FǮz�H     �  !�   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��         max       @v|          �  .T   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @*         max       @Q@           �  ;$   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @�V�         H  ;�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �(��   max       <�9X     H  =   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�6�   max       B4�     H  >X   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A��v   max       B4�^     H  ?�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >�I�   max       C���     H  @�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >�   max       C���     H  B0   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          {     H  Cx   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          =     H  D�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          3     H  F   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M��B   max       PR��     H  GP   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�U�=�L   max       ?ͿH˒;     H  H�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ���   max       <��     H  I�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?G�z�H   max       @F���R     �  K(   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��         max       @v|          �  W�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @(         max       @Q@           �  d�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @��          H  el   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         Bx   max         Bx     H  f�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�U�=�L   max       ?ͿH˒;     �  g�   
                     $       {               /      %                              	                           4   2            *      8      *                        
                  H            
         
   
                                 	         "   Nf�O��O�5O1�N���N 3�N�M�OJ
DPH|�P�9)O�Ng�O�i�N#{P9�iO<ʍO��N�%�N�NI׆N}-�NHHWN�o\N�\"NT5�N�a�O��N�!qO��N�O�Nq�kO6oIO���N�D�O=��P�ZP�N�=O)�HO�^�O��IO���P/�ONt,�P�O�(O _O2��O�pNa�OYAO�N�=(M��BO�*Nb��Nf;#N���Pa`�Or�O|M�N�ȾO��OJȞO���N�WN?��N�X�N�#�N�$�O��O��O.B,N&�N��N��O��N��)O�%vN�wO@sfN�s]<��<��
<���<�o<u<#�
<t�<t�;��
;�o%   %   %   %   �ě��ě���`B�#�
�49X�D���T���T����o��C���t���t���t��ě��ě��ě����ͼ�����/��`B��`B��`B��`B�����������+�C��C��\)�\)�\)�\)�t��t��t��t���P��P������w�#�
�'0 Ž49X�8Q�@��@��@��@��@��T���]/�ixսm�h�m�h�q���y�#�y�#��t����w���
���
��-���#)+)&)59:75,)	������������������������������������������������������������"$),)������	����������
#/<>CEGG<;/#

/9Ngt��������g[NB,5/���)>EPG)��������������������������������������������-9BO[]UV[lqm[OB6)X[hnriha[WXXXXXXXXXX���������&(
���������� 

������������������������������
�����������@HUanz�����zsna]UHE@�������������������������������������������������������������������������FIUXbn{{��{nnbUUIDFFV[gkrqhge[ZWVVVVVVVV��������������������������������������������� 

�������PZfht|��������tc[QLP	
#
���								35BNOWNHB50.33333333��������������������#0IbnsutwtnUI<4#����������������������������������������������)FWWO5�����@Hanz�������znaUJA?@ #(//<=>@<<6/'#     '6:>BKOW[]hsoh[OB6('#)5B[t������tgN5) ��������������������6COov���xmh\D6./-.36����"&�����������#//;/,##0IU^gnzn_UIGC5#9;;HQT[ahfeaXTHF>;89HLQTX\`aca_]TRNHCBDHyz�������������zwty����������������������������������������:IUbi{������{nbWD<6:����������������������������������������BDOX[\[OLBBBBBBBBBBBwz������������~zuwww_aimz���}ztma_______rt}���������~{wtrrrr"#&,/;<:62/$#"!!""""��)262364)�����xz}��������������{xx�����������������������������*0<EIU`^d_UI<40-++,*�����������������������������������������������������������������������������@BO[`hsphd[OJB@:@@@@�����



�����~����������������}~~��������������������������������������������������������������������������������NO[ht������xth[YOLFN��������������������X^ltx�����������re^X{�������������{{{{{{fk��������������tlgf��������������������/5<FHPRQTTRNKHD-('*/���������������������������Ⱥɺֺ���������ֺɺ��������=�9�2�1�5�=�I�V�X�b�d�d�b�\�V�I�=�=�=�=�����������������������������ʾ̾ʾ������Z�W�U�Z�]�b�g�s�����������������s�g�Z�Z���ݼ��������������������(�%���
���'�(�)�0�0�(�(�(�(�(�(�(�(�t�p�h�i�t������ŹŵůŶŹ�������������������������A�5�&��������A�Z�`�s�������~�g�[�A�'�!� �%�-�@�M������Ӽռ˼�������f�@�'���������������
�� ���
��������������g�`�Z�T�N�Z�f�g�s�z�|�s�g�g�g�g�g�g�g�g�`�G�;�2�,�4�;�G�T�`�y���������������m�`���������ĿѿӿѿϿĿ������������������������������������������#�8�F�O�L�;�/����g�e�Z�N�E�F�N�Z�g�s�����������������s�g�4�%�%�(�(�2�@�Y�f�r�{�}�~�|���~�f�Y�M�4ĦĜĚđėĚĦĳĺĿ����ĿĳĦĦĦĦĦĦ���������������������������������������żּӼʼȼʼҼּ�����ּּּּּּּ־f�f�f�i�s�}�����������������s�f�f�f�f�н˽ɽнݽ�����ݽннннннннн������������������������������������������������������������ʼѼּּؼּּҼʼ����T�L�P�T�`�m�y�}�y�n�m�`�T�T�T�T�T�T�T�T�L�K�D�G�L�Y�c�e�l�r�e�Y�L�L�L�L�L�L�L�L����������!�-�:�9�-�&�!�������������������������)�*�+�/�)�����������������������
��"�/�;�C�<�7�/�#�������������������������������������������a�_�\�`�a�m�t�z��}�z�m�a�a�a�a�a�a�a�a����޾�������"�(�(�"����	����!������'�-�:�F�_�l�x�b�_�R�F�:�-�!�6�.�)����)�,�6�B�K�O�R�X�O�B�6�6�6�6����������������������$�0�2�3�0�&���������ì×�H�?�E�aÇàì�����������ù����������������ú��'�$�����ܹϹ��H�@�<�/�$�-�/�<�H�U�V�a�g�a�_�U�H�H�H�H�ֹܹϹù��������������ùϹ۹�����ܿ�������������	����#�'� �!������������������������	��&�.�-�"�� ������ �.�;�C�G�T�`�m�y�{�m�a�T�G�;�.�"��Y�`�����¼����������ּ�����f�Y���������������������������������������Ž��x�r�|�������ݽ����(�=�7���н½��������������(�5�A�E�D�A�5�1�(��������y���������������������������������ݿۿѿĿÿĿĿ¿Ŀǿѿݿ����� ��������������������
��#�0�.�&�&�#����
����FF FFFF!F$F1F4F7F1F)F$FFFFFFF�������}�s�d�]�i�s�������������������������y�u�m�k�m�r�y�������������������������ݿѿѿοѿݿݿ�������������ݿݺY�Y�]�e�n�r�s�t�r�f�e�Y�Y�Y�Y�Y�Y�Y�Y�Y�H�D�=�8�8�>�H�T�a�z���������z�u�m�a�T�H������ƼƼƺ�������������������������������������������������������������������E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E��!��
�	��%�.�:�G�y�������̽ǽ��l�S�:�!���ܻͻû��ûлܻ�������������Z�M�9�/�)�*�4�A�M�Z�f�s�������{�n�h�f�Z����������������������� �������������ܻٻ߻����������� ������������
��
���#�0�<�B�`�b�i�b�V�I�<�#��j�V�I�=�2�0�=�I�V�b�oǈǖǜǝǛǔǈ�{�j�$���������������#�%�)�0�.�&�$������������������������������������������}��������������������������������²°¦¦¯²½������������¿²ŹŶŰŭŧŤŭŹ��������������������ŹŹ�b�`�_�b�h�n�r�{ŇŔŖśŜŔŇŅ�{�n�b�b�ֺɺ��������ɺֺ���!�4�:�-�&�����ֻF�;�C�O�S�_�l�x�~���������������x�l�S�F�m�b�a�_�a�i�m�z���{�z�r�m�m�m�m�m�m�m�m�l�k�m�s�w�w�x�x���������������������x�l�l�k�j�l�m�x�����y�x�l�l�l�l�l�l�l�l�l�l��
�������������
��#�/�<�H�R�N�H�<�/��C�8�6�*�"�)�*�6�C�N�O�V�O�K�C�C�C�C�C�CĦĚĆ�t�p�q�vāčęęĳķ��������ĿĳĦĿļĴĲĳĿ�������������������������ĿD�D�D�D�D�D�D�D�EEEE*E7E=E7E*EEED����������������������������������������� l . 1 6 : ] E  J  : ] A 0 L < # 6 � S R = H S E 6 U 9  - ; 7 f ? Z g ) V _ l  9 h < r ( o 7 1 X � 8 - n  b � t : C / v H A V V g @ ] - 9 d O M � m Q 1 X f X Q  �  2  4  1  �  m  �  �  ~  z  (  o  �  3  V  �  y  �  w  z  �  S  �  3  |  �  2  *  6  ~  {  |  �  �  �  &  �  �  �  �  	  r    �  /  N  �  �  0  �  _  b  �  9  c  �  �  �  �  x  �  �  Y  �  c  1  U  �  �  �  =  �  �  I  �  _  2  �    @  �  �<�9X;��
;ě�<t�;�o;��
��o������/����e`B���
�ě��D���T�����0 ż�j��t��u��t���C����
��/�ě��������\)�#�
�������]/�'e`B���
���-�,1�T���L�ͽ���D����E���w�����8Q�'<j�,1�D���L�ͽL�ͽ<j�0 ŽaG��'#�
�m�h��`B��O߽�7L�@��]/�q�����P�e`B�e`B�q�������u��O߽�E����P�}󶽗�P��7L�ȴ9�� Ž�;d��Q���پ(��B��Bh�B4�B�B�1B�B~$BNB	�(B Ba]B��B1CB��BqvB{cB!Y\B��B�oB!�(B /�B!�B��B'�B	�B�OB,B2B7B$tAB��B?�B&�?B�gB�B��B)B;�B hB�`B+/B1�xB-+KB�B&T}A�6�A��%B �FB��B�7B(��B*<eB*�B�JB'�A��'B
?gBH�B��B �B��B�0B&C�BCB�UB�1B9�B�Bt�BNBVB��BoTBDB�:B"EB
B
��B
�3B��B��BmeB�2BrB4�^B�B��B=�B�B9B	�B��BE_B�B=!B��BA�BHB!&B�GB��B!ǼB ?mB!ݭB��B'�B	<�B�VB,@B?mB@B$��B��BFHB'1ABA<B��BV`B�CB?[B;�B	8�BC^B1}�B-<�B��B&<sA��vA��B ��B@�B��B)>�B*]sB*?�B��B��A�tKB
:\B@eB��B?�BB"BCB&?�B��B��B�BFyB/zBA*B�BҗB��B�9B?XB�IB"��B
2)B
��B
�wB��B�HBv�@<MuBM�ALJ�A�]�A%`A5��A�]�A��A�D-@��yA�^#A�{�Ak��AxMA��2A��t@�'�A���Aϖ�A��AD�]A*��A�t�@�:�Aio�?�N!@a+�A�8�A��AJBgA�|tAZ_l@|=�A�E|B��A��m>���A��>�I�A[� A��Ad�@���A�b�A)J,A��A��QA}�A��C���A���Ap9�A�3I?��A�ZB&RB/tC�,8AlX@���A>�LA���@�/OA��.B��B	E.B0�A��TA�x�A�(RA��@KjS@�6�A���@�*@� �A��B �_AߏPA�?C�^<AJvy@C��Bx�ALP^A��2A��A5�}A�w?A��A�e�@�O^A�w�A��kAk�Ax��A���A���@�?qA���A��A΁AErA*�MA�"�@���Ai*?ז�@i��A�oA�rEAIUA�y/AY;�@s��A��B	<�Aɒ>��A�~>�A[�A�~�Ae��A��A�uWA!/JA� 'A�S�A}J`A��8C���A��An��A���?�)A�juB�jB2C�'�A	O@��A>��A��@��A�G�BJeB	A�B�A�Y�A�u�A�~5A�e�@D$a@���A�x9@���@� iA��AB ��Aߙ�A䣖C�`AJHh   
                     $   !   {               /      %                              	                           5   2            +      9      *                                          I            
         
   
                                  	         #                              3   9         %      5                                                      !         =   )               !   7      3                                          1                                       !                     !                                    !   '               /                                                               3   )                  +      1                                          /                                                                     Nf�N�YNw�ZO1�N���N 3�N�M�O+3�O��IP.�O�N1PO��N#{P��N�DO>эN�%�N�NI׆N}-�NHHWN�o\N�\"NT5�N"�O��N�!qO�	YN�O�Nq�kO I�O|z�Ns�O�NPFY(P�N+��N�ؐO� �OK��O���P	�Nt,�O���O�(O _O2��O�pNa�N�`iO�N�=(M��BO�*Nb��Nf;#N]<TPR��O��O|M�N�ȾN���OJȞO���NI�uN?��N�X�Nv�N�$�O��O���O.B,N&�N��N��O1�N��)Oa@�N�wN�;�N�s]  d  �  �  @  �  �  e  �  4  	  �  q  *  2  �  o  X  m    �    K  �  d  �    �  �    U  d  -  J  �  G  �  �    �  �      �  U  +  5  v    �  �  r  	  �  @  �  3  �  �    �  �  %  �  �  �  �  �    *  �  N  �  �  ^  �  c  X  �  �  w  �  m<��<�C�<e`B<�o<u<#�
<t�;��
�#�
�#�
%   ��o�t�%   �#�
�T������#�
�49X�D���T���T����o��C���t���1��t��ě����ͼě����ͼ�/�����+��w��`B�+�#�
�o���o�,1�C����\)�\)�\)�\)�t���w�t��t���P��P�����,1�49X�0 Ž0 Ž49X�@��@��@��L�ͽ@��@��]/�]/�ixսq���m�h�q���y�#�y�#���
���w��E����
�\���#)+)&	)56754)						������������������������������������������������������������"$),)������	���������� #/<<ACED<5/)#KNS[t}�������tg[TNHK���� $�������������������������������� ������������$*36BKNOVYYROBA63+)$X[hnriha[WXXXXXXXXXX�������
!%
������������

������������������������������
�����������@HUanz�����zsna]UHE@�������������������������������������������������������������������������FIUXbn{{��{nnbUUIDFFV[gkrqhge[ZWVVVVVVVV��������������������������������������������� 

�������MPR]hty��������te[SM	
#
���								35BNOWNHB50.33333333��������������������#<IcnrsrrrnbXI<9(����������������������������������������������5BOMF5!����@Hanz�������znaUJA?@"#//0<=<91/-#!""""""ABHOX[_hkhg[VODBAAAA%)5B[gt������gN5)!%��������������������56COlu��xmh\F6/0/05���������������#//;/,##0<R[cinxk\UJG?79;;HQT[ahfeaXTHF>;89HLQTX\`aca_]TRNHCBDHyz�������������zwty����������������������������������������[bdmn{���������{nba[����������������������������������������BDOX[\[OLBBBBBBBBBBBwz������������~zuwww_aimz���}ztma_______rt}���������~{wtrrrr"#(./49840/#""""""""����.30242)����yz�������������|zxy�����������������������������/0<AHISRIB<80/--////�����������������������������������������������������������������������������@BO[`hsphd[OJB@:@@@@��


 


 ���������~����������������}~~��������������������������������������������������������������������������������NO[ht������xth[YOLFN��������������������egkt~���������{tkdce{�������������{{{{{{rt��������������tmlr��������������������,/19<HLLPPMHH<8/,*,,���������������������������Ⱥɺֺ���������ֺɺ��������I�@�=�6�4�<�=�I�R�V�b�b�Y�V�I�I�I�I�I�I���������������������¾������������������Z�W�U�Z�]�b�g�s�����������������s�g�Z�Z���ݼ��������������������(�%���
���'�(�)�0�0�(�(�(�(�(�(�(�(�t�p�h�i�t����ŹŷŲŹŹ���������������������������A�>�5�0�#���
��(�A�I�d�k�h�h�g�`�N�A�@�4�.�-�4�@�M�r�������������������r�Y�@���������������
�� ���
��������������g�b�Z�Y�Q�Z�c�g�s�v�z�s�g�g�g�g�g�g�g�g�m�`�T�O�I�T�`�a�m�y���������������y�v�m���������ĿѿӿѿϿĿ����������������������������������������"�5�C�J�H�;�/��	���g�^�Z�S�N�L�N�R�Z�g�m�s���������w�s�g�g�Y�M�@�4�1�.�4�7�@�M�Y�f�j�m�r�v�r�n�f�YĦĜĚđėĚĦĳĺĿ����ĿĳĦĦĦĦĦĦ���������������������������������������żּӼʼȼʼҼּ�����ּּּּּּּ־f�f�f�i�s�}�����������������s�f�f�f�f�н˽ɽнݽ�����ݽннннннннн������������������������������������������������������������ʼѼּּؼּּҼʼ����T�L�P�T�`�m�y�}�y�n�m�`�T�T�T�T�T�T�T�T�Y�P�L�G�J�L�Y�^�e�e�f�e�Y�Y�Y�Y�Y�Y�Y�Y����������!�-�:�9�-�&�!�������������������������)�*�+�/�)�����#������������������
��!�/�9�A�<�6�/�#�����������������������������������������a�_�\�`�a�m�t�z��}�z�m�a�a�a�a�a�a�a�a�����߾�����	����"�&�&�"��	�	���!����!�)�-�:�F�S�_�h�f�_�Y�K�F�:�-�!�6�2�*�2�6�B�G�O�O�S�O�B�6�6�6�6�6�6�6�6���������������������$�.�0�$�!�����ùæÓ�z�n�Q�L�_�pÇàìú��������Źù����������������ú��'�$�����ܹϹ��H�E�<�;�<�E�H�K�U�a�c�a�Z�U�H�H�H�H�H�H�Ϲ͹ù����������ùιϹڹܹ޹ܹعϹϹϹϾ�������������	����&�� ���	���������������������	��� �"�'�#��	����"���"�.�<�F�G�T�`�m�w�z�m�_�T�G�;�.�"�������������ɼ�������
����ּʼ������������������������������������������Ž��{�t�x�������н����(�6�(���ݽн����������������(�5�A�E�D�A�5�1�(��������y���������������������������������ݿۿѿĿÿĿĿ¿Ŀǿѿݿ����� ��������������������
��#�0�.�&�&�#����
����FF FFFF!F$F1F4F7F1F)F$FFFFFFF�������s�r�h�a�g�o�s�����������������������y�u�m�k�m�r�y�������������������������ݿѿѿοѿݿݿ�������������ݿݺY�Y�]�e�n�r�s�t�r�f�e�Y�Y�Y�Y�Y�Y�Y�Y�Y�H�D�=�8�8�>�H�T�a�z���������z�u�m�a�T�H������ƼƼƺ�������������������������������������������������������������������E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E��:�!������.�:�G�y�������ȽĽ��l�S�:���ܻллȻлܻ���������������Z�M�9�/�)�*�4�A�M�Z�f�s�������{�n�h�f�Z����������������������� ���������������������������������������
��
���#�0�<�B�`�b�i�b�V�I�<�#��j�V�I�=�2�0�=�I�V�b�oǈǖǜǝǛǔǈ�{�j�$������	����!�$�(�&�$�$�$�$�$�$������������������������������������������}��������������������������������¦±²¿��������¿²¦ŹŶŰŭŧŤŭŹ��������������������ŹŹ�b�`�_�b�h�n�r�{ŇŔŖśŜŔŇŅ�{�n�b�b��ֺɺ��������ɺֺ����!�-�$������F�;�C�O�S�_�l�x�~���������������x�l�S�F�m�b�a�_�a�i�m�z���{�z�r�m�m�m�m�m�m�m�m�l�k�m�s�w�w�x�x���������������������x�l�l�k�j�l�m�x�����y�x�l�l�l�l�l�l�l�l�l�l���
���
���#�&�/�<�H�L�J�H�D�<�/�#��C�8�6�*�"�)�*�6�C�N�O�V�O�K�C�C�C�C�C�CĚėČĄ�|�v�}āčęĦİĳĸĽ��ĿĳĦĚĿļĴĲĳĿ�������������������������ĿED�D�D�D�D�D�D�EEEE*E*E,E*E$EEEE���������������������������������������� l ) - 6 : ] E  R  : k 4 0 E - * 6 � S R = H S E A U 9  - ; 6 c 5 H d ) + D o  8 S < v ( o 7 1 X B 8 - n  b � s : 9 / v < A V f g @ ` - 9 \ O M � m C 1 V f I Q  �  �  u  1  �  m  �  g  �  g  (  Q  ;  3  �  �  �  �  w  z  �  S  �  3  |  O  2  *    ~  {  _  U  q  6  �  �  C  �  �  �  R  �  �  �  N  �  �  0  �  1  b  �  9  c  �  �  �  �  C  �  �  �  �  c  �  U  �  �  �  =  5  �  I  �  _  H  �  �  @    �  Bx  Bx  Bx  Bx  Bx  Bx  Bx  Bx  Bx  Bx  Bx  Bx  Bx  Bx  Bx  Bx  Bx  Bx  Bx  Bx  Bx  Bx  Bx  Bx  Bx  Bx  Bx  Bx  Bx  Bx  Bx  Bx  Bx  Bx  Bx  Bx  Bx  Bx  Bx  Bx  Bx  Bx  Bx  Bx  Bx  Bx  Bx  Bx  Bx  Bx  Bx  Bx  Bx  Bx  Bx  Bx  Bx  Bx  Bx  Bx  Bx  Bx  Bx  Bx  Bx  Bx  Bx  Bx  Bx  Bx  Bx  Bx  Bx  Bx  Bx  Bx  Bx  Bx  Bx  Bx  Bx  Bx  d  }  �  �  �  l  V  =  "    �  �  �  |  Q  %  �  �  �  n  t  �  �  �  �  �  �  �  �  p  W  ;    �  �  �  z  O  &    �  �  �  �  �  �  �  �  �  �  �  �  �  �  w  d  P  9    �  @  ?  >  ;  5  .  %        �  �  �  �  �  �  �  	      �  �  �  q  S  5    �  �  �  `  :    �  �  8  �  �  4   �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  p  Q  2    e  \  P  B  4  #    �  �  �  �  g  ?    �  �  &  t  �   �  �  �  �  �  �  �  q  R  )  �  �  �  I    �  l    �  �  k  >  G  R  \  s  �  �    )  4  -    �  �  �  #  �  �  N  <  �  �  F  �  �  �  	  	  	  �  �  �  e  &  �  b  �    �  /  �  �  �  i  N  .    �  �  �  y  S  /    �  �  �  c  3    a  f  j  n  q  r  s  u  u  t  s  s  t  v  x  z  `  ?    �  �  �  �  �  �  �    )  )         �  �  �  �  \  $  
    2  5  7  :  <  ?  A  ;  2  (            �   �   �   �   �   �  �  �  �  �  �  �  �  �  �  �  h  <    �  m    �  e  �  '    ;  Q  a  j  n  m  d  T  ;    �  �  �  d  5  �  x  /  �  �    $  =  N  U  X  W  R  I  =  *    �  �  �  N    �  c  m  d  \  T  L  A  4  !    �  �  �    S  '  �  �  �  b  +      �  �  �  �  �  �  �  �  �  �  y  c  G  "  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �             �  �  �  �  �  �  �  �  r  X  >  &     �   �  K  A  6  ,  "    
  �  �  �  �  �  �  �  �  w  i  [  N  @  �  �  �  �  �  �  �  �  �  ~  y  u  p  l  g  d  `  ]  Z  V  d  `  [  R  I  =  /      �  �  �  �  �  h  H     �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  v  b  C  $    �  �  �  �  �  �  �  �     �  �  �  �  �  �  [  .    �  �  r  @    �  �  �  ~  u  l  b  W  I  :  (    �  �  �  �  �  V     �  �  �  �  �  �  �  �  �  �  �  �  �  j    �  �  S     �   �        �  �  �  �  �  �  �  w  e  Q  :    �  �  t     �  U  P  K  E  >  6  -  $        �  �  �  �  �  �  _  -   �  d  Y  N  C  :  1  (  "        �  �  �  �  x  G    �  �  %  *  '    	  �  �  �  �  �  �  m  M  .    �  �  �  �  �  F  H  J  H  F  =  +      �  �  �  �  D  �  �  [    �  �  d  s  �  �  �  �  �  s  a  J  )    �  �  �  �  |  �  �  �  �     3  D  >  *    �  �  �  K    �  �  G  �  �  G  �  �  A  �  �  �  �  �  �  �  r  N    �  �    �    �  �  8  �  �  �  �  �  �  g  9     �  n    �  h    �  :  �    -   �  �  �               �  �  �  �  �  �    j  U  ?  '    :  ]  h  �  �  �  �  �  �  �  �  �  �  n  D    �  �  t  4  �  �  �  �  q  \  E  *    �  �  �  �  �  c    �  (  �                �  �  �  `    �  a    �  _  �  |    �  {  ~  z  n  \  E  +    �  �  �  �  q  O  .  	  �  �  h   �  j  }  g  �  �  o  V  L  B  )    �  �  A  �  y  �  U  �  *  U  R  O  M  I  C  <  6  -  "      �  �  �  �  �  �  f  J  &  *  *    �  �  �  �  �  ]  &  �  {  a    �  o  �  b     5  /  )        �  �  �  �  �  �  �  o  U  8    �  �  �  v  u  t  s  q  o  n  g  ]  S  J  B  9  2  +  $              �  �  �  �  �  ~  s  i  d  S  <  !    �  �  ~  /   �  �  �  �  �  �  �  �  �  �  �  {  u  k  a  Y  Q  J  d  �  �  �  �  �  �  y  n  b  N  6    �  �  �  i  5  �  �  �  T            l  o  g  X  ?    �  �  �  |  N    �  �  o  +  	    �  �  �  �  �  �  �  �  �  l  N  .    �  �  �  @  �  �  �  �  �  �  �  �  �  �  k  S  :      �  �  �  t     �  @  8  0  )        �  �  �  v  K  !  �  �  �  n  ?    �  �  �  �  �  �  z  �  �  �  u  b  D    �  �  �  [    �  w  3  .  *  %  !          �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  u  h  \  O  C  �  �  �  �  �  �  r    �  x  #  �  i    �  ?  �  t    �            �  �  �  E  �  �  D    �  p     g  �  �   �  �  �  �  �  �  �  �  t  A  �  �  j    �  w    �  $  �  J  �  �  �  �  �  h  N  4      �  �  �  g    �  �  <  �  [  %              �  �  �  �  �  �  �  �  �  �  �  �  �  b  u  �  �  �  �  �  �  �  u  c  D    �  �  �  f  >     �  �  �  �  �  �  �  n  Y  C  .    �  �  �  �  s  L  #  �  �  �  �  �  �  �  �    �  �  �  x  J  
  �  Z  �  �  ;  �  �  v  y  |  �  �  �  �  �  �  �  �  �  �  u  N  $  �  �  �  W  �  �  �  �  �  t  V  4    �  �  �  k  >    �  �  �  T  #       �  �  �  �  �  �  �  g  F  !  �  �  �  {  I  	  �      !  E  |  �  �  �  �  r  _  N  ?  ,      �  �  �  �  �  �  �  �  �  �  {  q  d  U  F  2      �  �  �  �  {  d  N  N  ?  /      �  �  �  �  �  �  {  i  N    �  �  x  P  4  U  �  }  b  C    �  �  �  g  -  �  �  U    �  Q  �  I  �  �  �  �  �  �  �  �  �  {  }  v  ^  9    �  �  W    �  L  ^  S  H  >  3  )        �  �  �  �  �  �  y  ]  A  $    �  �  e  8    �  �  �  �  �  �  d  >    �  �  �    �   �  c  X  M  A  0      �  �  �  �  �  �  �  p  Q  *  �  �  8  K  Q  H  7  +  ?  W  M  A  +    �  �  R    �  B  �  @  v  �  �  �  �  �  �  �  �  �  �  �  �  �  v  `  F  *    �  �  �  �  �  �  �  �  �  �  �  �  Y    �  {    �  b    u  9  w  k  `  R  D  6  &    �  �  �  �      �  �  �  �  �  �  q  �  �  �  �  �  �  �  �  �  �  _    �  2  �  5  �    �  m  /  �  �  �  �  b  <    �  �  �  y  P  %  �  �  �  �  a