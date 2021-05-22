CDF       
      obs    N   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?��+I�     8  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       MԷ~   max       P��!     8  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ��j   max       =+     8      effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?8Q��   max       @Fc�
=p�     0  !T   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ���Q�     max       @vf�Q�     0  -�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @,         max       @M�           �  9�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @��         8  :P   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ��   max       <�1     8  ;�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�ѡ   max       B/��     8  <�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A��8   max       B0>�     8  =�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >U.<   max       C�K     8  ?0   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >AML   max       C�!�     8  @h   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          Y     8  A�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min          
   max          =     8  B�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min          
   max          =     8  D   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       MԷ~   max       P��!     8  EH   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�4�J�   max       ?��J�M     8  F�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ���   max       =+     8  G�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?8Q��   max       @F:�G�{     0  H�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�z�G�     max       @vf�Q�     0  U    speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @*         max       @M@           �  aP   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @��         8  a�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         @o   max         @o     8  c$   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?~Ov_ح�   max       ?��J�M     `  d\                        2            &      
      %         "         +                  "         #         *   9      )         +   4               
   2   	      X      	      
   	                  N                           
      	   W               >N^qRN-8hN�yO�N�H�O��N���P2�N�'O� 7O��wO��O��O�!O]X�PD�8OS�hOp�O�U>Na��N�dO�\�O)�mNr!�O���OσO$�O���N؎ O{�:O��N�@7OĘO�nP��!MԷ~O>��Od O1O�@�O��7N�f~N�GYN<�}N��`O6�/O�?Nݏ�N�T�OʝoP� N�C�N6O��N�>�OP;>O�oN1�O?��Ns��O��\N��:N��@N�CTN�RN�tN"�>N�-NXv+N�J9O�BN�q9O�3(N�x"OX)lM��N�<CO���=+<49X<o;o:�o�ě��o�o�o�o�t��t��#�
�49X�49X�D���D���D���T���e`B�u�u�u��C���C���C���C���C����㼬1��1��1��9X��j��j��/��h���������o�C��C��\)�\)�\)�\)�t��t�������w��w�',1�0 Ž0 Ž49X�49X�H�9�T���T���T���]/�q���y�#�}�}󶽁%��%��o�����+��t������� Ž�j���������������������������������������������������������'	������#&/363/#���������������������������������������������������������������������������������	"(#
�������U[gt����������tgaWTU��������������������#0IUbnzncZUI<7.$KNO[`gtt�����tg[NLKK #+.1<HUadfa[`UH/#! Zm�����������ma]a[Z����������������������

���������������������������MN[ggog][ONNLLMMMMMMJUabnyz�zvnaUPJJJJJJ����������������
#%+,)'#
����%%")5?DEDBHQNB5+[_gt������������tg][�������� ������������6COhv~~uqOC6��)5<>53)	8;HTadms}{�zmaTH;958������������������������������������������������������������;527;HTamqz{�~xjTH;�
#ERb{������{0#
���66BBCFEB646666666666O[htz~~|zztheb[YVROO����
!
�������������������������NW[t�����������t[QON����������������������������������������#/0::/,#��������������������CN[gnoqg][ZNGCCCCCCC���
#-/)#
��������&#�������������������������������
 	�������5BHNSUTRKB5)
HNU`nz���������naUIH���� ��������������������������������=BOV[^aca[QOBB<97===��������������������#0<ILQPPIKIF<0!)6O[goojdh[ZOJA6��������������������<FHUanvz}ynfaUHB<98<���������������������	)5:@BA>5)"�������	��������]glt~�����tthgb_]]]]EOP[hhnhg\[[[XODEEEE6BLN[^da[UNDB<976666DIUXWUTMIGGGDDDDDDDD���������������������������������������������������������������������������������������������������

##-/*#






��
(./78.3/-
���������������������������������������������;<HHINKH<9;;;;;;;;;;�����������������������������������������ݽٽ׽ݽ�����	��������ݽݽݽݽݽݾ����������������������������������������	��
��#�+�#���	�	�	�	�	�	�	�	�	�	�	�z�t�n�f�\�Y�\�a�i�n�q�r�z�ÇÐÎÏÇ�z�g�a�[�T�[�_�g�t�t�g�g�g�g�g�g���s�j�h�s������ʾ׾�������׾���������������!�#�!�!�����������������������������!�0�@�K�I�C�0�$����������!�������������5�(����(�5�=�N�Z�s���������s�g�Z�N�5����ܾپھ����	��"�&�*�&���	���𾾾��������������ʾ��������ؾҾʾ������������������μ��������ּʼ������������}�z�v�z�z�������������������������H�>�/�#������#�/�4�<�H�Q�S�\�W�U�H�Ŀ��ѿ׿��(�A�Z�g�s�y�p�b�U�Q�L�A��ݿ��<�7�8�2�1�<�F�I�U�b�k�n�v�y�r�g�b�U�I�<ŹűŭŪŭŭŹ������������������������Ź�y�`�U�K�L�T�`�m�y���������ƿĿ��������y¦§²³²¦�����	����#�(�0�/�-�(�����������������������ɺ���������ɺ��M�@�:�4�)�-�4�@�M�Y�r�z����y�r�k�f�Y�M��
������(�5�@�5�.�(��������T�P�R�Z�g�s�����������������������s�g�T����ۿֿҿɿſÿĿѿݿ���� �"� �������������y�x�y�������������ĿȿɿĿ������;�.�	����������	��)�.�C�H�I�S�Q�G�;�;�3�/�)�%�'�/�;�F�H�N�Q�R�H�;�;�;�;�;�;�������������������	��!��������������"� �� �"�+�/�;�H�T�a�i�m�l�a�T�H�;�/�"�H�?�<�8�0�3�<�H�U�\�\�_�^�U�H�H�H�H�H�H�������������������������������������������������ƶƭƫƱƳƽ��������������������|�N�A�8�/�1�=�g��������������������-�-�!�!�!�-�:�F�B�:�-�-�-�-�-�-�-�-�-�-�r�i�m�w����������ʼռмʼ�����������r�� �������������)�6�9�8�6�3�)�������������� �����	���"�(�(�$�"���	�����x�w�s�t�v�x�������û˻û���������������ݻܻܻ�����'�4�@�M�V�T�=�4�'��������������������������������������������;�1�/�'�.�/�;�H�S�T�]�Z�T�H�;�;�;�;�;�;�@�<�6�@�L�Y�]�Z�Y�L�@�@�@�@�@�@�@�@�@�@�������������	���������������y�o�m�e�g�m�z�����������������������������'�4�@�E�\�b�d�_�Y�M�@�'���������������+�'� ����������T�H�J�T�Y�`�m�t�y�|�}�y�s�m�d�`�T�T�T�T�������������*�C�O�\�b�e�d�Z�O�6���m�a�H�;�1�/�2�H�S�j�m�z�������������z�m�s�k�n�s��������������������s�s�s�s�s�s���������������������������������������޿T�K�N�T�V�`�m�y���������������y�m�`�T�T�h�a�_�c�h�tāčĘėčĈā�t�h�h�h�h�h�h���������������������нݽ��ݽսνĽ����������|�{�������������������������������������������ĽɽĽĽ����������������������	��������������	��!�.�4�9�3�.�(�"��	�����z����������������������������������������ýòòù���������������)�����ččā�āĄčĚĦĳĽĿ��ĿĺĳĦĚčččċčĐĚğĦĳĿ��ĿĿĳĳĦĚčččč�ù����������ùϹϹܹ���ܹ۹Ϲùùù��/�)�/�0�8�<�H�U�_�a�c�a�a�U�H�<�/�/�/�/�_�Y�X�_�l�v�x�������x�l�_�_�_�_�_�_�_�_��ܹ�������������������軪�����������û��������������������������ʼȼʼּ�������������ּʼʼʼʼʼ��I�F�?�I�U�b�n�r�{�{�{�n�b�U�I�I�I�I�I�I�0�-�#��
������	�
��#�0�1�8�=�=�<�6�0�������������ûлջлл̻û�������������D�D�D�D�D�D�D�D�D�D�D�D�EE4E;E.EED�D�������
��������������������������J�G�P�Y�b�r�~�������������������~�r�M�JE�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E������������
��#�'�*�(�#��
��������������ڼռ׼�������� ���������� ` 8 C d & ? K S B : # F K 9 8 Z % 3 ( T W 0 2 W l V H a N Q = P 6 * b h h D i U W ^ 6 1 1 J # @ b & A   _ / < C \ \ 1 F L H W D 9 u b n c H @ 5 [ ' v `  O  �  C  8  g  �  �  �  �  (  �  �  .  �  M  �    �  K  �  �  �    v  �  ~  0  q  �    -  �    4  �  T    �  T  v  h  =  �  �  ^  �  �  �  �  �  �  �  �  \  2  �  �  �  I  �  �  l    �  �    K  <  n  �  �  U  �  �    $    �  K<�1;�`B:�o�t��D����h�e`B�m�h�D����h�+�<j��P��1�ě��D���C���h�<j��t����
�ixս\)��j���ͽ0 ż��ͽP�`���ͽC��e`B�#�
����o�����o��\)�D���49X���P����8Q��w�'49X��1�0 Ž49X��󶽋C��<j�'H�9�H�9�}󶽉7L�@���o�L�;J����u���P��7L�y�#��hs��C���7L��t����㽕�����\������-����B5fBzbB$IB<]B��B!�@B+��B��Bt�Bm�B	�}B�2B&سB	LB�B ��B�RB�CB+.�B��BSUB"J{B$��B5�B�B ZBc�B/��BagA�9]B�B ��B[A�ѡB&�2B+�B��B{�B ��B��B ��B�6B�7B!z�B�B�iBF�B!q�BrAB�B�uB�B"�B��BN�B%�=BM5B"b~B�6B ��Bm�B�qB	�(B)�BKB&߰B�B�B-nB�_BY�B$֥BܷBw�B#�BB�`B!B@/BD+BAmB>�B��B!�kB+�>B�VBN1B@WB	˂B��B&�*B	BAB {�B�B��B*��B�aBC�B"GB$��B?�B"�B
FBI[B0>�B��A���B=�B �B>�A��8B&ҹB?�B��B��B ��B�B ҼB�~B�kB!��B��Bd�BAB!��B@�B>�B�B��B%�B��BU`B%�B��B"B3B��B �B�^B�_B	��B?4B@KB&��B�XB<YB->?Bs�B�OB$��B��BC�B@�B�BÎB�KA-ڛAHc�A�v-A��*A���AK3@X$B	3A���A�3oAZ>PAQ:/@���A���A�2`A�c�A�T,A�{Aoy�A���A���@7F�@�y�A��A�`A�Asw7A^s�A�w�A��A���Aĵ�A��B�A���@w�@�A�='A��|@��k@�g�A�/�A���?�"A.�*A���@�F�?dV�Ai�1B eA���A�ǇA��YAk��A���A#=A�ʜA$�LA\�A�^A�(NA�E{A�4\>U.<A�P+@�D�?1\_@���ApA�9jA�V�@��C�SA�P@��C�KA���A�|A,��AG�!A���A�ñA��*AJ��@c��B	��A���A���AZ�AQe�@�P�A��ATA���A틁A��*AoA�O�A�~�@8�e@�>�A�|A��fA�p`Au�A[!�A�|�A�?EA���A�z�A��MB�>A�s@t�S@�
AՅBA�b�@��@�$A�lA���?��kA/�-A�/Q@�Q�?g��Ai�B =�A�c�A��BA���Al��A�ɅA"�UA�}*A$�XA\�=A��A�yoA�S�A��@>AMLAă�@�2�?"x@��FA��A�;KA�M�@���C�HsA�}*@��C�!�A��A                        3            &            %         "         +         	      	   #         $      	   *   9      *         ,   4               
   2   	   	   Y      	         	                  O      	                     
      
   W               >         
               +      !      %   !         1         !         )            #      '                     =                                                -                  #                                                %                        
               %      !                  #         !                                                =                                                                  !                                                               N^qRN-8hN�yN�mN&��O^��N���O��N�'O� 7O
�O�h�O�O�!O]X�O�|(O�Op�O�U>Na��N�dO�]N���Nr!�O\�xO���O$�O�5�N؎ O{�:N��NHPuN�%sO�nP��!MԷ~N���Od N��'O`��O��N�f~N���N<�}N��`N��O�iNݏ�NtdsO"��O��N�C�N6O��Ny��O+YO�:ON1�O��Ns��O9�N�#NT�N*�N�RN�tN"�>N�-NXv+N�J9O�BN�q9OU�YN�x"OIu&M��N��O���  �    �  �  b  [  @  ;  m  F  z  k    S    �  r  8  �  D    d  �  �    �  �  Y    �  �  f  M  >  �  p  �  �  r  �  	�  {  K  =  i  �  *  �  |  �  �  �  �  N  �    �  �  �  �  �  '  l  �    P  )  �  A  g  V  �  `  
  �  M  �  �=+<49X<o%   �D���T���o�49X�o�o��C���t���t��49X�49X��1��C��D���T���e`B�u�t����㼋C���t����㼋C���/���㼬1����h�ě���j��j��/�,1���+��w�@��o�\)�C��\)��P���\)��P���P�L�ͽ���w��w�49X�49X�49X�0 ŽD���49X��\)�]/�]/�q���]/�q���y�#�}�}󶽁%��%��o��1��+������������j�����������������������������������������������������������������#-/30/#��������������������������������������������������� �������������������������������	"(#
�������cgt���������tig_[cc��������������������-1<IUbkibb[UQI<;740-KNO[`gtt�����tg[NLKK #+.1<HUadfa[`UH/#! kmz������������mhdfk����������������������

���������������������������MN[ggog][ONNLLMMMMMMJUabnyz�zvnaUPJJJJJJ����������������������
#((%%##
���%%")5=BDBB@E?5.+ckt������������tg_]c�������� ����������*6COV[^\OC6*)5<>53)	8;HTadms}{�zmaTH;958������������������������������������������������������������;527;HTamqz{�~xjTH;�
#ERb{������{0#
���66BBCFEB646666666666U[httxyvtihg`[ZVUUUU����
!
�������������������������V[ht���������thd[VUV����������������������������������������#./88/&#��������������������CN[gnoqg][ZNGCCCCCCC����
"$#
�������$!����������������������������������������)59BFJIGB?5)&^cnz����������zna[[^���� ��������������������������������=BOV[^aca[QOBB<97===��������������������"#0<JPNNJIA<:0#)6@O[ennih[ROJB6��������������������;<@HKU`nronha_UHH<;;�������������������� )57;;:55*)	��������������agt|���~tmgdaaaaaaaaJOV[ahjhb[OLJJJJJJJJ6BLN[^da[UNDB<976666DIUXWUTMIGGGDDDDDDDD���������������������������������������������������������������������������������������������������

##-/*#






���
#&'&$ 
���������������������������������������������;<HHINKH<9;;;;;;;;;;�����������������������������������������ݽٽ׽ݽ�����	��������ݽݽݽݽݽݾ����������������������������������������	��
��#�+�#���	�	�	�	�	�	�	�	�	�	�	�z�v�n�h�a�^�[�^�a�k�n�y�zÇÌÊÉÇ�z�z�}�t�g�c�g�h�t�������x�s�q����������ʾӾվӾʾ¾�������������!�#�!�!���������������
�����������������0�>�I�H�A�0�$���������!�������������5�(����(�5�=�N�Z�s���������s�g�Z�N�5�����������	���"�"�"����	�����ʾ����������������ʾ׾���������оʼʼ������������ʼּּ��������ּͼ��������}�z�v�z�z�������������������������H�>�/�#������#�/�4�<�H�Q�S�\�W�U�H���߿�ݿ����5�A�N�S�T�R�L�F�A�5����<�:�<�?�9�<�=�I�U�^�b�n�o�s�n�k�b�^�U�<ŹűŭŪŭŭŹ������������������������Ź�y�`�U�K�L�T�`�m�y���������ƿĿ��������y¦§²³²¦�����	����#�(�0�/�-�(�������ɺ������������ɺֺ޺��������ֺɼY�S�M�A�@�4�4�7�@�M�Y�f�r�s�|�u�r�g�f�Y��
������(�5�@�5�.�(��������g�Z�S�T�Z�d�s�����������������������s�g��ݿؿԿʿǿƿѿݿ������ ������꿟�������y�x�y�������������ĿȿɿĿ������;�.��	���������	��"�.�9�>�@�A�E�@�;�;�3�/�)�%�'�/�;�F�H�N�Q�R�H�;�;�;�;�;�;�������������������	��!��������������H�A�;�/�1�9�;�H�T�Y�a�a�a�`�T�K�H�H�H�H�H�F�=�<�9�<�H�U�V�Y�W�U�H�H�H�H�H�H�H�H�������������������������������������������������ƶƭƫƱƳƽ��������������������|�N�A�8�/�1�=�g��������������������-�-�!�!�!�-�:�F�B�:�-�-�-�-�-�-�-�-�-�-��s�t������������������������������� �������������)�6�9�8�6�3�)����	���� ��	����"�&�%�"���	�	�	�	����y�w�w�z������������������������������	�����������'�4�@�K�G�@�4�.�'�����������������������������������������;�7�/�(�/�0�;�H�P�T�W�T�H�B�;�;�;�;�;�;�@�<�6�@�L�Y�]�Z�Y�L�@�@�@�@�@�@�@�@�@�@�������������	�����������������|�z�s�m�k�m�z�����������������������
���'�4�@�C�O�[�a�b�]�Y�M�@�'���������������+�'� ����������`�Z�T�M�T�\�`�m�y�}�y�r�m�a�`�`�`�`�`�`�*��������*�6�C�M�O�U�R�O�C�C�6�*�T�H�<�8�;�C�H�T�a�m�z�����������{�m�a�T�s�k�n�s��������������������s�s�s�s�s�s���������������������������������������޿T�K�N�T�V�`�m�y���������������y�m�`�T�T�t�k�h�c�h�h�tāĊčđčăā�t�t�t�t�t�t�����������������������Ľʽнҽн̽½��������|�|�������������������������������������������ĽɽĽĽ������������������������	����������	���"�,�.�4�/�.�#�"������z������������������������������������������þ�������������������������ĚĐčĂċčĚĦĳĸĿ��ĿķĳĦĚĚĚĚĚĎēĚĥĦĬĳĹĳĮĦĚĚĚĚĚĚĚĚ�ù����������ùȹϹֹӹϹùùùùùùù��/�)�/�0�8�<�H�U�_�a�c�a�a�U�H�<�/�/�/�/�_�Y�X�_�l�v�x�������x�l�_�_�_�_�_�_�_�_��ܹ�������������������軪�����������û��������������������������ʼȼʼּ�������������ּʼʼʼʼʼ��I�F�?�I�U�b�n�r�{�{�{�n�b�U�I�I�I�I�I�I�0�-�#��
������	�
��#�0�1�8�=�=�<�6�0�������������ûлջлл̻û�������������D�D�D�D�D�D�D�D�D�EEE%E*E1E*E%EEED�������
��������������������������L�J�R�Y�d�r�w�~�������������������~�Y�LE�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E��
����������
��"�#�%�#���
�
�
�
�
�
��ڼռ׼�������� ���������� ` 8 C ^ C ; K P B : % 3 8 9 8 6 / 3 ( T W . 4 W p U H U N Q  T 2 * b h B D c R E ^ % 1 1 I ! @ \  /   _ / +  _ \ & F > F @ 0 9 u b n c H @ 5 , ' s `  O  �  C  8    X  �  �  e  (  �  ,    Z  M  �  E  G  K  �  �  �  e    �    �  q  U    -  �  s  �  �  T    �  T  $  �  D  �  �  ^  �  !  J  �  �  Y  8  �  \  2  y  q  m  I  )  �  �  �  k  G    K  <  n  �  �  U  �  �    �    �  K  @o  @o  @o  @o  @o  @o  @o  @o  @o  @o  @o  @o  @o  @o  @o  @o  @o  @o  @o  @o  @o  @o  @o  @o  @o  @o  @o  @o  @o  @o  @o  @o  @o  @o  @o  @o  @o  @o  @o  @o  @o  @o  @o  @o  @o  @o  @o  @o  @o  @o  @o  @o  @o  @o  @o  @o  @o  @o  @o  @o  @o  @o  @o  @o  @o  @o  @o  @o  @o  @o  @o  @o  @o  @o  @o  @o  @o  @o  �  �  �  �  �  �  w  \  B  B  =  4    �  �  b  *  �  �  x          �  �  �  �  �  �  �  �  �  �  y  n  c  Y  N  D  �  �  �  n  Y  D  -    �  �  �  �  �  ^  7    �  �  �  c  |  x  �  �  �  �  �  �  �  �  ~  h  O  0    �  �  y  M  #  u  �  �    .  J  e  z  �  �  �  �  �  �  �  {  v  r  o  l  N  Y  Z  Y  Z  Z  Y  P  D  5    �  �  �    H  �  �  b  %  @  4  (        �  �  �  �  �  �  �  y  i  Z  J  :  *    �  8  1      �  �  �  �  �  �  �  �  ~  <  �  N  �  ~  �  m  g  a  Z  T  K  8  %    �  �  �  �  �  �  }  h  R  <  '  F  E  ?  -    �  �  �  �  �    !      �  �  6  �  8   }  �  �    5  W  k  w  y  s  c  J  $  �  �  �  H  �  �    �  �  �  -  Z  h  k  Z  >  "    �  �  �  �  p  D    �    �  �  �  �  �  �        �  �  �  �  �  �  ~  )  �  =  �   �  S  M  F  >  5  )         �  �  �  �  �  �  �  u  a  X  Q    �  �  �  �  �  �  �  v  \  ?  "    �  �    �  }    �  x  y  �  �  �  �  �  �  �  �  �  f  .  �  �  f    �    �  C  U  c  m  q  m  _  F  >    �  �  c  %  �  �  �  �  �  �  8  0  "    �  �  �  �  o  I    �  �  k    �  y  C    �  �  �  �  �  �  �  �  �  �  �  �  �  x  Y  .  �  �  $  �   �  D  I  O  T  Z  [  S  J  A  9  -        �  �  �  �  q  O      
    �  �  �  �  �  �  �  �  �  �  �  s  b  S  D  5  �  �      0  :  A  L  [  d  b  Y  @    �  �  c    �    ~  �  �  �  �  �  �  �  �  j  O  '  �  �  �  A  �  �  t   �  �  �  �  �  �  y  n  `  P  A  /      �  �  �  �  z  Y  8    	      �  �  �  �  �  �  �  �  �  �  l  ]  Q  C  0    �  �  �  �  �  �  v  a  E     �  �  �  f  K  �  �  >  �  >  �  �  �  s  c  Q  ;  &  
  �  �  �  �  o  Q  R  ]  _  R  D    ,  :  H  Q  W  Y  U  Q  G  @  .    �  �  I  �  |  �  �       �  �  �  �  �  �  �  �  �  �  �  �  �  m  G     �   �  �  �  �  �  �  �  �  m  M  -  
  �  �  �  Y    �  �  X  /    r  �    ,  I  `  r  ~  �  {  i  N  #  �  �  j    �  �  �      )  :  G  T  c  f  X  ;    �  �  9  �  s    �  -  C  F  I  K  L  L  J  H  >  4  &      �  �  �  x  K     �  >      �  �  �  �  x  J    �  {  %  �  R  �  R  �  �  �  �  �  �  Y    �  �  |  >    �  �  �  F  �  �    �  �   �  p  h  _  V  L  <  +    	  �  �  �  �  !  �  V    �  s  *  y  u  m  �  �  �  �  �  �  �  �  n  %  �    e  �  �  ~  Y  �  �  �  �  �  �  o  Y  E  /    �  �  �  �  �  �  �  �  �  _  \  T  U  p  ]  F  /      �  �  �  q  A    �  �  &  �  �  �  �  �  �  �  �  �  r  G    �  �  :  �  Y  �  �  w  N  �  `  �  	  	R  	t  	�  	~  	j  	G  	  �  �  6  �  �  *  �  L  r  {  m  `  R  D  6  (    
  �  �  �  �  �  �  e  F  +    �  F  I  K  I  E  =  5  ,      �  �  �  �  �  �  }  q  c  U  =  ?  A  D  E  B  @  >  6  &      �  �  �  l  ?     �   �  i  ^  S  H  <  0  $      �  �  �  �  �  �  �  v  X  :    x  �  �  �  �  �  �  �  �  }  s  h  [  P  I  A  1       �    (  %      �  �  �  `  :  	  �  t    �    �  (  �  �  �  �  �  �  �  �  �  �  �  s  `  N  <  #  
  �  �  �  j  6  ^  j  w  x  r  j  [  L  8  #    �  �  �  8    �  �  �  U  	�  
�    �  �  7  [  u  �  }  W    �  =  
�  	�  �    �  �  �  �  $  J  a  v  �  �  �  �  Y     �  �  "  �  b     �  <  �  �  �  �  �  �  x  g  W  G  6  &      �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  }  w  q  N  ;  (    
  �  �  �  �  �  �  u  Y  ;    �  �  �   �   J  d  n  x  �  �  �  �  �  �  �  �  s  c  I  /  
  �  �  6  �  l      �  �  �  �  �  �  �  �  c  6  �  �  j    �  <   �  �  �  �  �  }  h  O  /  	  �  �  �  Z  %  �  �  w  "  �  F  �  �  �  �  �  �  }  q  e  Z  M  ?  1  #       �   �   �   �  p  v  �  �  �  �  �  p  V  6    �  �  �  Z    �  �      �  �  �  �  �  �  u  b  J  2    �  �  �  �  m  F     �   �  �    U  �  �  �  �  �  i  0  �  w  �    B  
c  	m  d  9      $  &      �  �  �  ?  �  �  ?  �  �  $  �  ]  �  �  d  g  e  c  e  i  i  \  N  @  2  #      �  �  �  �  �  n  B  .  �  �  �  �  �  �  �  �  �  �  y  .  �  6  �    �  �  U    �  �  �  �  �  �  �  �  �  n  H    �  �  {  F    �  �  P  P  O  O  N  N  M  L  L  K  M  O  R  U  X  [  ^  `  c  f  )      �  �  �  �  �  �  b  ;    �  �  �  T    �  �  `  �  �  �  �  �  �  �  �  ~  l  P  +    �  �  �  w  P  *    A  ,      �  �  �  �  �  �  {  l  ]  N  >  /       �   �  g  R  <  &    �  �  �  �  �  v  ^  I  3      �  �  �  �  V  P  E  5  #    �  �  �  �  �  j  E    �  �  �  H    �  �  �  {  }  �  �  �  �  �  �  o  Z  ?  !  �  �  �  �  c  8    7  J  Z  `  ^  R  >  1    �  �  4  �  2  
|  	�    �  4  
      �  �  �  �  �  e  ,  �  �  /  �    i  �  �  :  o  �  �  �  �  �  g  @  %    �  �  T    �  �  m    �    �  M  >  0  !      �  �  �  �  �  �  �  �  |  l  \  L  <  ,  G  W  f  q  |  �  �  p  Y  ?    �  �  �  [  %  �  �  B  �  �  �  [    �  �  M  �  �  ,  
�  
@  	�  	?  �  �  >  �  �  �