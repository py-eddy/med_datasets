CDF       
      obs    G   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?��n��P       �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M���   max       P�c�       �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ����   max       =]/       �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?@        max       @FL�����       !    effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?���P    max       @v������       ,   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @)         max       @R            �  70   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @ˣ        max       @�F�           7�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ��P   max       =D��       8�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A���   max       B0�       9�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�,   max       B0"�       ;   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >�T�   max       C��E       <0   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >�e   max       C�'�       =L   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �       >h   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          ;       ?�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          6       @�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M���   max       Pc�;       A�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�ݗ�+j�   max       ?�o hۋ�       B�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��1   max       =]/       C�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?@        max       @F,�����       E   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�(�X    max       @v������       P(   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @,         max       @R            �  [@   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @ˣ        max       @��           [�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         >%   max         >%       \�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�$tS��N   max       ?�k��~($     0  ^      /   	                     
            #      	   '      	                        �                        E      	   
                              +            
                  !      #         $      !                  NA��O��"N�$�Nۈ�N��\N��O��fN vPNK>�N
��NGEP#��O)��P��OI��N��P�c�N�u�N�o�Ob��N�.N�»NgO	N�[eN��3N7��P�gO��OW��O�6N���O���O��)O{Pc�;NHܖN��N_��N�ݵNqЩNk��N�8sNu��N/:OĚGN�{�Nb҅O���Nh��N!��O&�N�84O��Nϴ�Om$O7`�NM.�O�e�N>��O�'�O>GO��O�dCO[�)O��N$r'M���O�;N��N:7O��=]/;D��:�o��o��o�D����o���
�ě��ě��t��t��t��t��49X�49X�D���D���e`B�e`B�u��o��C���C����㼛�㼛�㼬1��9X��9X�ě����ͼ��ͼ��ͼ�/��/��/��/��h�����o�\)�t��t����',1�,1�,1�49X�49X�8Q�8Q�<j�@��@��D���T���T���aG��ixսq���y�#�����7L��7L��O߽�O߽��-����)36)"������������������������������������������������������������()6BEKKFB6)("&((((((����	


���������� 
#+.'#
� ������������������������qt|�����ttnlqqqqqqqq��������������������9BOSXOONB=9999999999[at������������th^Z[���

��������*CO\eelghaTC6����������������������#02466<@<50&#T`������������m[TJBTyz�����������|zwryy?BLNS[\dfgf[ZNMDB9??ggt������������xtnmg��������������������MQRR[dhikoqmhhgb[ZOM$)/588;5)`afnzz�~|zzqnha^XY``�����������IN[gqig[NKIIIIIIIIII�����	((���������������������������#/HUagj]UH</-#����
 
�������;;@GHHIKLKJH;757;;;;����������������������������������BNO[hjtvyutjh[OGFBBBan���������������ziaGHTaagaaTOLHGGGGGGGG"#./;<GA<4/#""""""#./770/$#'0<@IKUbnoonbUJI<0/'�����������������$)-052)&"?BHO[hkkhjkh[[POCB??[[htvwvvthgc[Z[[[[[[����������������5BQVSRNKB5)stu�������ttssssssssq{}�����������{uqqqqKQ[gt����������tfNIK)0<>@DINNIH<90/())))HITUXZ[UUIEDHHHHHHHH����������������������������������������[t�����������tcZTPP[�����������������������
�������5BINgtz�������tgREB5����	
!
�������������
�������cgt~ttgc]cccccccccc������������������������������������������������������������#'/<anx�}xnaUH<0.!��
#)(/+#
 ��������$.-*"������45BBNONIBB:5//444444�����������������������	�������!�����������������������./:<EHU^cWUNHD</-,+.�ֺκӺֺ������ֺֺֺֺֺֺֺֺֺֻB�:�.�3�<�F�I�_�l�x�����������}�x�l�S�B�!����!�-�:�F�S�X�_�l�d�_�S�F�:�-�!�!�O�L�B�9�B�O�[�h�t�{āćā�t�h�[�O�O�O�O������#�/�<�A�E�=�<�/�#�������s�k�f�^�Z�Z�Z�e�f�s�t���������s�s�s�sùìáàÕÎÕàù�������������������ù�ּϼʼȼʼּ������ּּּּּּּ��/�)�/�5�<�H�U�Z�U�U�H�<�/�/�/�/�/�/�/�/�U�R�H�C�<�H�U�V�W�V�U�U�U�U�U�U�U�U�U�U�a�X�]�a�n�zÁ�z�r�n�a�a�a�a�a�a�a�a�a�a��׾ž¾����������׾���������	���㾌�����}�}������������ʾ˾ʾ���������������۾¾������ʾ���	�"�.�8�<�B�B�;�"������������(�4�M�Z�\�V�M�A�4�(�������������ûлܻ���������ܻл̻û���ݿ������������>�G�C�7�7�B�N�Y�N�(�������������������������������������ҿѿͿĿ¿Ŀ̿ѿݿ������������ݿѿ����
������(�5�A�E�E�A�?�5�1�(������������������������������������нĽ������������Ľнݽ��������߽ݽҽ������������������������������������������������x���������������������������������������������������	���������������������������	�
���	�����������������������ӻӻ޻��!�4�M�Y�f��������r�B�4�����������������	��"�,�%�#�"���	�������6�*�(�(�)�(�%�)�6�B�E�T�d�^�[�U�Z�O�B�6�������������������Ǽʼμּ���ʼ����������������������������������������������I�7�4�,�'�0�<�I�U�b�n�{�~łŅł�n�b�U�I���������������������������������������������������������Ŀѿӿݿ���ݿѿǿĿ��ֺ������r�M�D�B�I�X�r���ɺ���	�
� ������
���"�/�/�1�/�"����������5�4�)�'�)�)�5�B�D�N�R�X�N�B�5�5�5�5�5�5�<�8�<�=�H�U�a�b�a�a�U�H�<�<�<�<�<�<�<�<�-�&�!�!� �!�#�%�-�-�:�;�F�F�F�F�B�:�:�-�t�m�l�t�t�t�t�t�t�t�t�t�t�t�t�m�t�xāčĚĦĪĦĚđčā�t�t�t�t�t�t�Ϲ˹ù������ùϹܹ�����������ܹ۹Ϲ�����������������������������������������ŔœŐŔŞŠŭŲŭŬŧŠŔŔŔŔŔŔŔŔ������������������*�6�@�D�A�6��������¿´²¦ª²¿����������¿¿¿¿¿¿¿¿������ݽֽн˽нݽ������������������������¿���������������'�4�9�#����佞�������������ĽнѽнǽĽ��������������(�%����(�4�A�A�F�A�4�(�(�(�(�(�(�(�(����žŹŸŶŹŹ���������������������������������*�6�>�6�0�*��������ĦģģĦħĹĿ��������������������ĿĳĦ���������������������	����������������ؿ����޾�����	���"�.�8�;�>�7�.�"������������� �������	����������������������������������������������ּ������������ʼ�����.�:�>�:�.�!������������������������������������������������������������%�2�4�.������T�I�D�=�3�4�=�D�I�O�V�b�h�{��}�{�o�b�T�Y�P�L�I�J�L�Z�e�r�����������������~�r�YE�E�E�F
F#F-F3F=FVFoF|F�F�F�FsFZF=F1FE�D�D�D�D�EEE&E*E7EBEPEQENE>E*EEED�D�úðìàÝàìù����������� �&������ú�a�a�U�U�T�U�V�a�n�o�z�}�z�n�a�a�a�a�a�a�ѿϿпѿݿ����߿ݿѿѿѿѿѿѿѿѿѿѽY�S�G�;�P�S�`�l�y�����������������y�l�Y�����������
��!�-�.�:�.�!��������������!�!�!�������������������E�E�E�EEuEsEuE�E�E�E�E�E�E�E�E�E�E�E�E� M 1 S l  R L 9 P � L b + S F ^ H ( P C T b ; D H T , @ d S � & M   Q ; W & p 6 t > L P < ' � ( x ] 3 H / M K q y U < & \ ! z t 0 ~ h d o a c    k  7      �  �  !  J  m  �  M  x  l  �  �    �  �    �  �    z    �  t    [    S  �  �  �  K  �  h  �  s  :  �  �  
  �  ]  �  �  �  �  �  >  g  �  �  �      �  i  L  �  N  >  �    :  q    �  �  <  d=D���49X�o�o�o�#�
�ě���`B�t���o��C���h�ě��0 ż�`B���
�L�ͼ�j��j��󶼛�㼴9X��9X�o������9X��P��h���'�`B�,1��w�\)�ě���h�t���P�'t����D����w�'q���8Q�P�`��{�@��<j�e`B�Y���\)�aG�����ixսT�����T�q���� Ž�7L������ ŽƧ�hs��\)���T�������T���B�BgB,�3B?B�B^�B�B!�MB|�B 0�Bp8B 6�B$�B0�B �B%@NB4B�pBv#B
�aB�CB�_B��B�XB8B�iB"B}RB^�B$"�A���BָB�SB[�B�jA��BOB�B&��B�B��B{B��B��B=B�"B)#�B	�OB&0ZB'$_B��B�B
�B�hB�!B	d�B$9B-=`B	�nB��B��B!o�BhzB�BB�9B�uB��B:B!�lB�B�B�;B,�dBB�,BP�B��B!v�B�B ��BB�B YB$8OB0"�B ��B%?�B7�B�/BD�BC�B��BA,B�B�{B:�B�&B=�B�kBL3B#��A�,BхBǆBx�B�}A��<BB�BǻB'A�B ~B��B7�B߅B�^B8B�sB)��B	�ZB%¥B'xB�BE@B
;BÖB��B	�{B$�B-ӠB	C�B�kBP�B"(`BSBAB�B�VB�B��B@B!ŲB��@@9�@�g�@A�2A�E�AB�!A�}eA��A�
�A���A�C�AV�AJP�A[&�A7u�@�KHA�9=Aк�A}��A�P#A��A(�NA��A��B��A���@��A�F�A�4j@��_A�n�A�XA���Ax�I@
�A�1BA�W(A��!@w�8A�w@A�SV>�T�A��dA�ZA��yA��/A,��A���A$��A8 A�}�A�9�A�s�A��A\sAYCA�`�ATwA�jA�ڊB)�?��>C��EC�v�AЏA�!fA}>�AD�A6\@^�4C�@C�@���@|TA�R,A�x�AB�A͂{A8iA�x1AĀ�AƃAR.AK�AZ�A7+�@�<IA�+AЀ,A~�A��*A�@A&x�A���A�}�B�aA�l�@��mA���A�_�@�n�A�|AA���Ax��@��A�6A��A��@s�A�yoA�}.>�o�A���A�uA���A���A+lA�w\A"z!A9$�A�~�A��A�^�A��A\��AZjA���A�A��fA�}qBC^?���>�eC�MA��AƀnA}�EA
A	�W@\�C�'�      0   
                        	         #      
   '      
                        �   	                     F      
                                 ,            
                  "      #         $      !                                                      /      )         ;                              5                        6                              !         %                              )      !         )      %                                                      /               +                                                      6                                                                     )               )      %                  NA��O\�)N�$�Nۈ�NC��N��O��fN vPNK>�N
��NGEP-�O
��O?��O�N3VqP%��Nm��N�o�OH��N�.N�»NgO	N�T�Nq�MN7��O���O��O�N��N���O���O��)N��;Pc�;NHܖN��N_��N�ݵNqЩNk��NP�Nu��N/:O�6%N�{�Nb҅Ok�wNh��N!��O&�N�84O���Nϴ�O`��O
��NM.�O�e�N>��O�f�N�R�O��O�dCO[�)O��N$r'M���O�;N��N:7O��    �  U  �    �  x    �  6  �  �    �  _  �  �  �  p  j  �  (      �  `  
�  s  p  �  �    �  L  �  �     �  
  �  �  X  n  �  �    O  :  j  k  6  �  e  8  �  �  E    �  �  �  �  r  S  �  a     �  @     �=]/�ě�:�o��o�D���D����o���
�ě��ě��t��#�
�49X���ͼe`B�e`B�ě��u�e`B�u�u��o��C���t����
���㽬1��1���ͼ����ě����ͼ��ͼ�`B��/��/��/��/��h������P�\)�t������'e`B�,1�,1�49X�49X�@��8Q�@��H�9�@��D���T���y�#�e`B�ixսq���y�#�����7L��7L��O߽�O߽��-����)36)"������������������������������������������������������������%)*368;BEBA6,)%%%%%%����	


���������� 
#+.'#
� ������������������������qt|�����ttnlqqqqqqqq��������������������9BOSXOONB=9999999999]ct������������th`[]����

������*36CKOSQMB6*��������������������#/0100#_gnw������������ma\_vz��������}zvvvvvvvv?BLNS[\dfgf[ZNMDB9??pt�������������{tpnp��������������������MQRR[dhikoqmhhgb[ZOM$)/588;5)aagnxz~|zzxnnma_ZZaa�������������IN[gqig[NKIIIIIIIIII����������������������������������������#/<HTSIHA<2/#��

����������;;@GHHIKLKJH;757;;;;����������������������������������IOX[`hrtutoh[OOHIIIIan���������������ziaGHTaagaaTOLHGGGGGGGG"#./;<GA<4/#""""""#./770/$#'0<@IKUbnoonbUJI<0/'�����������������$)-052)&"LO[gg_[ONDLLLLLLLLLL[[htvwvvthgc[Z[[[[[[����������������5BMTQQPLIB5) stu�������ttssssssssq{}�����������{uqqqq[gt{��������tgc[WRT[)0<>@DINNIH<90/())))HITUXZ[UUIEDHHHHHHHH����������������������������������������R[dt����������td\VRR�������������������������
������EN[gtw������tsg[VNIE����	
!
�������������
�������cgt~ttgc]cccccccccc������������������������������������������������������������#'/<anx�}xnaUH<0.!��
#)(/+#
 ��������$.-*"������45BBNONIBB:5//444444�����������������������	�������!�����������������������./:<EHU^cWUNHD</-,+.�ֺκӺֺ������ֺֺֺֺֺֺֺֺֺֻF�;�:�B�F�P�_�l�x�������������x�l�_�S�F�!����!�-�:�F�S�X�_�l�d�_�S�F�:�-�!�!�O�L�B�9�B�O�[�h�t�{āćā�t�h�[�O�O�O�O�/�#�#����#�+�/�7�<�B�<�8�/�/�/�/�/�/�s�k�f�^�Z�Z�Z�e�f�s�t���������s�s�s�sùìáàÕÎÕàù�������������������ù�ּϼʼȼʼּ������ּּּּּּּ��/�)�/�5�<�H�U�Z�U�U�H�<�/�/�/�/�/�/�/�/�U�R�H�C�<�H�U�V�W�V�U�U�U�U�U�U�U�U�U�U�a�X�]�a�n�zÁ�z�r�n�a�a�a�a�a�a�a�a�a�a��׾Ǿľ����������׾������ ���	���㾘�����������������������Ǿž����������"��	������ܾ����	��"�*�/�/�.�&�"��	�������(�4�A�M�V�P�M�A�4�(���û����ûǻлܻ��ܻлǻûûûûûûû������ݿοǿ����ݿ���!�-�$�'�*�4�4�(����������������������������������������޿ѿͿĿ¿Ŀ̿ѿݿ������������ݿѿ������	���(�5�?�A�D�D�A�;�5�/�(������������������������������������нĽ������������Ľнݽ��������߽ݽҽ������������������������������������������������z���������������������������������������������������������������������������������	�
���	�������������������������������'�4�@�M�Y�_�b�`�Y�M�4�����������������	��"�,�%�#�"���	�������6�1�*�*�,�.�6�B�O�O�[�a�[�W�R�O�B�A�6�6�����������������ļɼɼ����������������������������������������������������������I�7�4�,�'�0�<�I�U�b�n�{�~łŅł�n�b�U�I���������������������������������������������������������ĿǿѿؿڿѿпĿ��������ֺ������r�M�D�B�I�X�r���ɺ���	�
� ������
���"�/�/�1�/�"����������5�4�)�'�)�)�5�B�D�N�R�X�N�B�5�5�5�5�5�5�<�8�<�=�H�U�a�b�a�a�U�H�<�<�<�<�<�<�<�<�-�&�!�!� �!�#�%�-�-�:�;�F�F�F�F�B�:�:�-�t�m�l�t�t�t�t�t�t�t�t�t�t�t�t�m�t�xāčĚĦĪĦĚđčā�t�t�t�t�t�t�ù����ùϹܹ�ݹܹϹùùùùùùùùù�����������������������������������������ŔœŐŔŞŠŭŲŭŬŧŠŔŔŔŔŔŔŔŔ���������������*�6�>�B�?�6�*��������¿´²¦ª²¿����������¿¿¿¿¿¿¿¿������ݽֽн˽нݽ�����������������������������������
�����
��������ؽ��������������ĽнѽнǽĽ��������������(�%����(�4�A�A�F�A�4�(�(�(�(�(�(�(�(����žŹŸŶŹŹ���������������������������������*�6�>�6�0�*��������ĳĩĦĥĥĭĳĿ��������������������Ŀĳ���������������������	����������������ؿ��	����߾����	���"�.�7�<�6�.�"�������������	�������	���������������������������������������������ּ������������ʼ�����.�:�>�:�.�!���������������������������������������������������������(�+�$����������b�W�V�I�E�A�I�R�V�b�f�o�{�~�|�{�o�k�b�b�Y�P�L�I�J�L�Z�e�r�����������������~�r�YE�E�E�F
F#F-F3F=FVFoF|F�F�F�FsFZF=F1FE�D�D�D�D�EEE&E*E7EBEPEQENE>E*EEED�D�úðìàÝàìù����������� �&������ú�a�a�U�U�T�U�V�a�n�o�z�}�z�n�a�a�a�a�a�a�ѿϿпѿݿ����߿ݿѿѿѿѿѿѿѿѿѿѽY�S�G�;�P�S�`�l�y�����������������y�l�Y�����������
��!�-�.�:�.�!��������������!�!�!�������������������E�E�E�EEuEsEuE�E�E�E�E�E�E�E�E�E�E�E�E� M ' S l 7 R L 9 P � L c ) F ? J D 5 P > T b ; C D T  @ E   � & M % Q ; W & p 6 t : L P G ' � ! x ] 3 H * M M l y U <  # ! z t 0 ~ h d o a c    k  �      \  �  !  J  m  �  M  O  1  �  T  V  �  �    �  �    z  �  �  t  M  [  X  �  �  �  �  �  �  h  �  s  :  �  �  Y  �  ]  �  �  �  �  �  >  g  �  _  �  �  d  �  i  L    �  >  �    :  q    �  �  <  d  >%  >%  >%  >%  >%  >%  >%  >%  >%  >%  >%  >%  >%  >%  >%  >%  >%  >%  >%  >%  >%  >%  >%  >%  >%  >%  >%  >%  >%  >%  >%  >%  >%  >%  >%  >%  >%  >%  >%  >%  >%  >%  >%  >%  >%  >%  >%  >%  >%  >%  >%  >%  >%  >%  >%  >%  >%  >%  >%  >%  >%  >%  >%  >%  >%  >%  >%  >%  >%  >%  >%          �  �  �  �  �  �  �  �  �  q  b  S  C  3  "    �  �  �  �  �  �  �  �  �  o  J    �  �    r  �  �  �   �  U  O  I  B  <  8  4  ,  "      �  �  �  �  �  {  a  V  K  �  �  �  �  �  �  �  {  m  ^  O  ?  1  #  %  =  U  d  o  {  �  �  �        	                  2  F  G  ?  8  �  �  �  �  �  �  �  �  �  �  {  i  T  ?  &  	  �  �  �  h  x  x  p  a  K  C  H  7       �  �  �  g  4     �  �  '   q        �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  6  B  P  j  {  e  H    �  �  |  E    �  �  W    �  �  L  �  �  �  �  �  �  �  �  �  �  �  �  s  b  Q  +  �  �  �  l  �  �  �  �  �  �  �  �  p  Y  ?    �  �  �  �  �  o  )   �  �  �      �  �  �  �  �  �  �  t  [  D  ,    �  �  =   �    M  s  �  �  �  �  �  �  �  �  �  x  J    �  �  Y  �  i  :  M  X  ^  ^  ^  \  W  N  ?  (  
  �  �  �  p  ;     �  S  �  �  �  �  �  �  �  �  �    
    �  �  �  �  �  f    �  R  i  �  �  �  �  �  �  �  �  �  U    �  �  �  ~  7    �  �  �  �  �  �  �  �  �  �  �  �  y  L    �  �  �  U     �  p  c  V  9    �  �  �  �  t  U  6    �  �  �    ;  s  �  a  i  j  h  f  d  a  ]  W  I  ,    �  �  �  t  K  %  �  �  �  �  �  }  x  q  h  _  U  L  >  )      �  �  �  �  j  H  (  %  !          �  �  �  �  �  �  �  �  �      	           �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �        
    �  �  �    	    �  �  �  �  r  C    �  �  �  �  �  �  �  �  �  �  �  �  o  [  C  +    �  �  �  �  n  `  ^  [  X  V  S  Q  X  c  o  {  �  �  �  �  �  �  �  �  �  E  �  	4  	�  	�  
*  
j  
�  
�  
�  
�  
�  
o  
   	�  	  @  �  E  h  s  h  ]  R  G  <  2  (    	  �  �  �  �  �  �  �  �  �  �  N  :  K  c  k  ]  J  5    �  �  �  x  M  #  �  �  �  �  Y  �  �    f  �  x  o  Z  @  %  	  �  �  �  g  .  �  �      �  �    w  p  g  Z  L  ?  2  3  C  S  c  t  y  |  ~  �  �      �  �  �  �  �  �  ]  3    �  �  �  �  �  ~  ]  P    �  �  �  �  q  |  �  �  ~  q  ^  F  +    �  �  �  �  ~  Z  *  /  4  :  @  G  K  I  F  B  =  7  .  "    �  �  �  �  �  �  �  �  d  H  ,    �  �  y  ,  �  m  �  d  �    q  #  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �     �  �  �  �  �  �  �  �  o  V  =  %    )  0    �  �  �  �  �  �  �  �  z  h  S  >  )    �  �  �  �  �  B  �  �  ]  
    �  �  �  �  �  �  �  q  Z  C  (    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �        &  �  �  �  �    x  m  `  T  E  7  (    �  �    �  +  �  �  O  :  "  ,  A  L  Y  `  b  ]  P  =  &    �  �  �  a    �  n  b  V  J  =  0      �  �  �  �  �  �  �  �    m  \  K  �  �  �  �  �  �  �  �  �  �  �  �  p  _  N  =  ,      �  |  �  �    t  b  K  0    �  �  �  �  \  6    �  �  {  �                �  �  �  �  �  �  �  ~  T  +   �   �   �  O  I  A  6  ;  �  �  �  �  z  g  T  *  �  �  �  S    �  �  �  �  �  !  2  9  :  5  (    �  �  �  I  �  �  H  �    R  j  ]  P  D  8  4  /  *  !      �  �  �          �  �  k  h  d  a  ]  Y  P  G  >  5  ,  #      	  �  �  �  �  �  6  0  *  "    	  �  �  �  �  �  �  }  V  -    �  �  �  \  �  �  �  �  w  h  X  H  7       �  �  �  x  R  ,    �  �  N  ^  d  Z  O  >  '    �  �  �  �  r  D    �  �  r  W  $  8  6  4  5  4  +  #      �  �  �  �  �  �  W    �  �  b  �  �  �  �  �  �  {  f  G  !  �  �  [    �  u  �  g  �   �  n  �  �  �  �  �  �    m  U  9    �  �  �  `    �  �  �  E  <  3  *      �  �  �  �  �  �  n  P  3    �  �  �  �    �  �  �  j  .  �  �  h    �  �  +  �  f    �  s  .  �  �  �  �  �  �  |  l  X  D  +    �  �  �  �  �  �  �  m  W  �  �  �  �  �  �  �  �  �  p  >     �  o    �  ;  �    �  �  �  �  �  �  �  d  H  -    �  �  �  �  j  6  �  �  w  !  �  �  �  �  �  k  Q  >  *    �  �  �  y  D  	  �  �  9  �  r  #  �      �  }  ?    �  �  H  �  �  _    �  m    �  S  7  
  �  �  f  O    �  �  y  w  �  �  p  F    �  �  �  �  �  �  �  j  H     �  �  d    �    ;          4  b  a  <    �  �  �  �  �  �  �  �  �  z  `  E  )     �   �   �       �  �  �  �  �  n  R  6    �  �  �  �  �  k  L  .    �  �  t  d  S  @  ,      �  �  �  �  �  e  0  �  p     �  @  6  ,  !      �  �  �  �  �  �  �  p  U  ;  �  �  R  �                          ;  c  �  �  �  �  �  �  �  �  �  v  ;  �  �  {  5  �  �  {  P  U  :  �  �    �  '  �