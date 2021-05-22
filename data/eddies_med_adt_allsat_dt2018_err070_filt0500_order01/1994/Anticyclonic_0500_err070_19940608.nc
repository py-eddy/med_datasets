CDF       
      obs    3   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?pbM���   max       ?�E����      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       N[   max       P�8      �  x   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ��   max       >t�      �  D   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?(�\)   max       @F
=p��     �      effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?��
=p�    max       @vm�Q�     �  (   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @0         max       @P@           h  0    effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�_        max       @�`          �  0h   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ��1   max       >k�      �  14   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�q�   max       B/�      �  2    latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�z�   max       B/��      �  2�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ?��   max       C�>�      �  3�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ?�w   max       C�F�      �  4d   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �      �  50   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          9      �  5�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          1      �  6�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       N[   max       Pd��      �  7�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�����A   max       ?��+j��g      �  8`   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��h   max       >t�      �  9,   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?.z�G�   max       @F
=p��     �  9�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�         max       @vm�Q�     �  A�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @*         max       @P@           h  I�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�_        max       @�{           �  JP   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         ?e   max         ?e      �  K   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?������   max       ?��Y��|�     �  K�   	                                 #      
      6      A   5   (   	                     	   	               9      	      !         C      	      �      $      $      Ng�pN��CNlbN�hO� �N��"N��N@fO:PN[O�PXP��O2�O=��N�?�O�G8O��P�8O���P&;VN;�0P&�N���N7��O���Ns��N�dvN�fOJ�AN�l1N-)�O
s�Oz�?O���O���N� �O�pXO�fN1RN��+P5N�N��O��O��N(�O=B�OD�jODC�NKsVN�$����o�#�
�o%   ;ě�;ě�;�`B;�`B;�`B<o<#�
<#�
<49X<T��<T��<e`B<u<u<�t�<�t�<�t�<��
<ě�<���<�`B<�<�=o=o=C�=C�=�P=�P=��=��=��=��=��=��=�w=<j=P�`=ix�=ix�=��=��=��-=ě�>t�>t�limt�������tllllllllPMMTU[ajnqvwpnaaUUPPXV[gt}tg[XXXXXXXXXX����������� ������!)1BN[gqsomigd[N5!����� �������qtwz������������tqq)05652)�������

������������������������������������
$%#����������������������.(%&',/<HRUafYUF<2/.��������������������\Yanz�����znkca\\\\��������

�����
#/3<?A</#�������.21)$�������������������������.*5[g�����������mNB.#).,$#)+8HUaz������znUH</)������ ������������������������������qompt{������������hq$ $*/6>B:6*$$$$$$$$����������������������������������������������()23.)��gcehtu~������tphggggDCKNV[`][NDDDDDDDDDDWZ[\cgt��������ztg[W�����)6@B@6�����������������������)5BMC6'��')5753)(! )5BN[^agptmgNB1(zvt����������������z  #04;60#          )*+03<?INOI=<0))))))��������������������PNTUVaemaUPPPPPPPPPP��������������������.,,/:AHTVW\aeheYH;2.�������
"#
�����$)2,)&1./15:BO[_def[VOB961����)6:61,)��������������������������������������������������������������غL�Y�e�n�h�e�Z�Y�X�L�F�E�L�L�L�L�L�L�L�LE�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E��������������������������������������������������ûȻлԻлŻû������������������Ŀѿݿ����!�$�#�������ڿѿƿ����Ŀ����������ĿǿĿ������������������������"�)�/�;�H�I�N�I�H�;�4�/�&�-�&�"� ��"�"��������������ھ������������������������������������������������������������¿²¯²²¿�����������������˾s����������ʾ���ʾ�����������t�n�s������B�U�[�j�[�T�C�6��������������������������������������������������������������������������������������/�<�B�@�>�=�<�/�#�!�����#�$�/�/�/�/����(�A�Z�a�q�z�s�f�Z�(����������m�y�������}�m�V�G�;�.�!��"�.�;�G�T�`�m����/�H�n�y�v�z�T�;�	������������������ù����������������ùÓÇ�z�u�t�}ÉÓêù�g�t�z�w��g�B�)�������)�U�^�g�������������������������A�Z�s�������Y�L�O�5�(���	���#�"�#�A�/�;�D�H�R�L�H�;�/�"�����"�,�/�/�/�/���(�2�/�(���������������������Ľ˽нӽֽͽĽ��������y�z�y�y�����`�m�y���������y�m�`�^�]�`�`�`�`�`�`�`�`�����������(�,�*�(�"������������������������������������������������������uƁƎƚƠƧƩƩƫƧƦƚƎƁ�|�u�p�l�r�u�S�_�l�s�l�l�_�[�S�M�F�B�F�@�F�R�S�S�S�S�;�G�T�[�T�Q�G�;�9�6�;�;�;�;�;�;�;�;�;�;�`�m�q�y���������������y�u�m�g�`�Z�V�\�`�����������׾߾��������׾ʾ���������������������ļü�������������p�]�R�T�g���#�0�I�V�_�d�h�k�j�b�U�I�<�#�� ��������������������������������������������������������������������������������������������������������s�g�Z�N�I�G�K�Y�w���ּ������ּ̼Ҽּּּּּּּּּּ����������������������������������������nÇÓÙâïõïÞÓ�z�a�H�0�0�<�A�J�a�n��(�4�5�4�-�(�����������������!�-�2�:�B�:�-�!�����������#�<�I�U�b�q�x�n�b�I�<�0�#��
�����
�D�D�D�D�D�D�D�D�D�D�D�D�D�D�DyDrDrD|D�D���������������������������������������������ûлܻ����	�����ܻȻû����������L�[�e�~�����������������~�z�p�e�Y�K�F�L������������������������������������}��E7ECEPE\E_EfE\EPECE@E7E1E7E7E7E7E7E7E7E7EuE�E�E�E�E�E�E�E�E�E�EyEuEpEuEuEuEuEuEu , I H J E ; U / O c D F 3   B G K ' X Z , I @ J ) - J [ 2 m 1 F { ! A X T ) ) 6 , D n R 0 h W � t V O    y    �  	  �  �    A  a  U  �  �  �  �     �  4  �  �  �  Q  �  �  S  #  }  �  `  �  �  K  >  �  �    �  b  �  B  �  B  7  �  q  �  `  �  U  )  }  ���1;�o��o<e`B<�h<#�
<49X<49X<49X<T��=\)=8Q�=�P<�1<�/=�+<�h=��
=�7L=m�h<���=49X<���<�`B=0 �=o=C�=�w=#�
=��=��=L��=]/=�v�=�O�=<j=�+=�\)=0 �=8Q�=��=H�9=q��=���>k�=��w=�;d=�E�>+>"��>(��B}�BD�B	U�B#<KB6eB��B
�]B��Bx}BˣBs�B"�B�BsqB�B#��B�B.�B!�B
��B��B`�B��B�"B0�B/�BO�B)�B�<B~4BS�B	�/B��B"��B�B\jB��B��B%�B&#4B�6B�B*K�A�q�BBɾB��BG�B�	BY)Bv�Bo�B<;B	9yB#9zBB3B B
��B�|B��B�>B?vB�EB�BkB5�B#�4BO�B�nB"@PB
�aB� B�dB��B�BH�B/��BzkB@!BnBBAWB	�8B��B"�@B�Bh�B�B�gB%��B&@�B�]B�WB*�HA�z�B>[B��B�B�/B $B@NB��?��C�>�A�5@���A�,nAt�9A��AAV�*A�85A��\AM.!A�8�A҈�A�ΗA¼�A7��Af�AA� `A���A���A�9A���A��_A��,A!ψAk��A3A�%B�y@��`Ad��Al��AR��@��;A�^	A�*A�ZVA��bA��@��8A�4^A6Zo@kǈA��C���@��o@��e@ �@��C���C��@?�wC�F�A�p{@���A�c�AtW�A�g�AV�yA��A��AN�A��kA҃�A�{�A��A6:�AiC?A�K2A˛	A�{�A��7A�x�A�6�A��]A$QAk�A3�A��B�7@��AcC�AmzAR�g@�b�A뀝A䅙A���A�z�A�P@��AȊ1A6$Y@s��A�%rC��{@�h�@���@�y@���C���C��   
                                 $            6      B   5   )   	                     
   	               9       	      !         D      	      �      %      %                                       !   +            #      9   !   -      +                                 !   !   !                  %                                                                  %                  1      %      '                                       !                                                N?�	N��CNlbN�?O�N��"N��NN@fO:PN[OA�O��AN���O7zNM�Ogv�O��Pd��Ot�O�4KN;�0Oڵ�N���N7��O���Ns��N�dvN�fOJ�AN�l1N-)�N<��O?��O���O�8�N� �O+�uOpA`N1RN��+O��N�Nv��Oy��ONN(�N���OD�jO1��NKsVN�$  ]  �  �    �  :  5  �  �  f  �  �  �  �  �    �  �  �  �  �  �  A  �  �  �  >    �  e  +  �  �  �  E  �  O  
  �  W  	Q  h  \  �  ^  �  �  R  	l  e  ��h��o�#�
��o<D��;ě�;�`B;�`B;�`B;�`B<��
<�t�<��
<T��<�t�<�`B<e`B<�`B=\)<���<�t�<�j<��
<ě�<���<�`B<�<�=o=o=C�='�=�w=8Q�=�w=��=<j=<j=��=��=q��=<j=T��=u=�l�=��=�-=��-=ȴ9>t�>t�njpt�����tnnnnnnnnnnPMMTU[ajnqvwpnaaUUPPXV[gt}tg[XXXXXXXXXX��������� ��������556;BFN[bdd`^[NHB>65����� �������{}��������������{{{{)05652)�������

�����������������������������������

��������������� ��������--/1:<@HQUWUKH</----��������������������a]anzz���}znmaaaaaaa�������

�����
#/3<?A</#��������(,+(#������������������������TR[at������������raT#).,$#-/<Uanz������znUH<3-������ ������������������������������qompt{������������hq$ $*/6>B:6*$$$$$$$$����������������������������������������������()23.)��gcehtu~������tphggggDCKNV[`][NDDDDDDDDDDfcglt����wtgffffffff�	')6=@=64)����������
�����������)5@KB5&���')5753)(&))5BLNX[[_`[NB85/(��������������������  #04;60#          )*+03<?INOI=<0))))))��������������������PNTUVaemaUPPPPPPPPPP��������������������...2>EHTZ^acfbVH;50.�������

�����$)2,)&A7:?BHO[^``[XOCBAAAA����)6:61,)��������������� �����������������������������������������������غL�Y�e�k�f�e�Y�L�G�G�L�L�L�L�L�L�L�L�L�LE�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E��������������������������������������������������ûŻϻûû������������������������������������ݿؿӿѿϿѿݿ꿟���������ĿǿĿ������������������������/�;�E�H�L�H�F�;�1�/�+�/�0�/�(�$�/�/�/�/��������������ھ������������������������������������������������������������¿²¯²²¿�����������������˾����������ʾվվʾþ����������������������)�B�L�U�O�G�B�6�)���������������������������������������������������������������������������������������/�<�?�=�<�<�2�/�-�#��"�#�/�/�/�/�/�/�/���(�4�A�Q�_�_�Z�M�A�4�(����������m�y�������}�m�V�G�;�.�!��"�.�;�G�T�`�m�	��/�@�U�a�g�^�H�"�	�����������������	àìù����������ùìàÓÇÆÀÆÇÓÙà�[�g�n�t�~�{�g�B�5�)��	�	���)�B�[�������������������������A�Z�m���}�g�P�E�A�5�(�����%�*�*�5�A�/�;�D�H�R�L�H�;�/�"�����"�,�/�/�/�/���(�2�/�(���������������������Ľ˽нӽֽͽĽ��������y�z�y�y�����`�m�y���������y�m�`�^�]�`�`�`�`�`�`�`�`�����������(�,�*�(�"������������������������������������������������������uƁƎƚƠƧƩƩƫƧƦƚƎƁ�|�u�p�l�r�u�S�_�l�s�l�l�_�[�S�M�F�B�F�@�F�R�S�S�S�S�;�G�T�[�T�Q�G�;�9�6�;�;�;�;�;�;�;�;�;�;�m�y�����������y�v�m�i�k�m�m�m�m�m�m�m�m���ʾ׾۾����������׾ʾ����������������������������������������r�d�Z�]�m��#�0�I�U�^�c�g�k�i�b�U�I�=�#������#���������������������������������������������������������������������������������s���������������������s�g�Z�Q�N�R�Z�g�s�ּ������ּ̼Ҽּּּּּּּּּּ����������������������������������������a�n�zÇÓÖãèäÓÇ�z�a�T�H�E�M�Q�U�a��(�4�5�4�-�(�����������������!�-�0�:�@�:�-�!����������#�0�<�I�U�b�h�a�I�<�7�0�#��
����
��#D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D~DD�D�D�����������������������������������������ûлܻ�����������ܻԻлǻûûûûûúL�[�e�~�����������������~�z�p�e�Y�K�F�L����������������������������������������E7ECEPE\E_EfE\EPECE@E7E1E7E7E7E7E7E7E7E7EuE�E�E�E�E�E�E�E�E�E�EyEuEpEuEuEuEuEuEu - I H L 2 ; J / O c / C ( $ \ < K " ; 6 , T @ J ) - J [ 2 m 1 C 1  A X : & ) 6 $ D h E  h A � s V O    O    �  �  S  �  �  A  a  U  *  �  �  #  �  �  4  �  B  G  Q    �  S  #  }  �  `  �  �  K  `  �  %    �  s  �  B  �  /  7  �  �  �  `  �  U  �  }  �  ?e  ?e  ?e  ?e  ?e  ?e  ?e  ?e  ?e  ?e  ?e  ?e  ?e  ?e  ?e  ?e  ?e  ?e  ?e  ?e  ?e  ?e  ?e  ?e  ?e  ?e  ?e  ?e  ?e  ?e  ?e  ?e  ?e  ?e  ?e  ?e  ?e  ?e  ?e  ?e  ?e  ?e  ?e  ?e  ?e  ?e  ?e  ?e  ?e  ?e  ?e  T  X  \  ^  _  [  U  L  A  4  %    �  �  �  �  v  O    �  �  �  �  u  h  U  <    �  �  �  i  6  �  �  r  ,  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  p  _  r  }  ~  z  r  f  V  @  #  �  �  �  T    �  �  E  �  �  E  �    E  c  ~  �  �  �  �  �  �  �  o  J    �  [  �  F  u  :  *      �  �  �  �  �  �  �  h  M  2     �   �   �   �   �  ,  .  1  3  5  2  .  +  &          �  �  �  �  �    Y  �  �  �  �  �  �  �  �  �  �  o  R  6    �  �  �  �  �  r  �  �  �  �  �  �  �  �  �  �    z  u  p  k  q  {  �  �  �  f  d  b  _  Y  R  K  G  D  B  ?  =  :  5  ,  #    
  �  �  �    0  U  v  �  �  �  �  �  �  �  �  Z  #  �  �  -  �  �  ^  |  �  �  �  �  �  �  �  n  D    �  �  g  '  �  �    �  �    -  U  {  �  �  �  �  �  y  O  "  �  �  v  6  �  �  �  k  s  {  �  �  �  �  x  p  i  \  H  3      �  �  �  �    �  �  �  �  �  �  �  �  �  �  s  K  �  f    �  o    �  v  l  �  �  �  �         �  �  �  �  �  E  �  X  �    6  @  �  �  u  f  W  G  :  %  
  �  �  �  �  m  O  3  	  �  m   �  1  �  �  �  �  �  �  �  �  i  S  >    �  �  3  �  (  .   Q    .  7  7  O  �  �  �  �  �  z  2  �  l  �  v  �  <  �  5  =  6  \  �  �  �  g  E  !  �  �  �  ]    �  �  9  �  &     �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  q  a  O  =  +  �  �  �  �  �  �  �  �  �  w  N    �  �  �  J  ;  �  �   �  A  9  1  (      	  �  �  �  �  �  �  �  �  |  h  K  .    �  �  �  �  �  �  x  n  e  [  Q  G  =  3  )        �  �  �  �  �  �  �  �  �  �  �  �  �  �    q  _  L  4    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  y  p  c  T  E  6  '  >  5  +  !          
        �  �  �  �  �  �  �  �    �  �  �  �  {  ^  9    �  �  �  ]  .  �  �  �  i  5    �  �  �  �  z  m  a  T  E  6  %      �  �  �  �  �  t  R  e  ]  U  M  M  O  P  H  <  /  "      �  �  �  �  �  �  �  +  (  &  #                 �  �  �  �  �  �  �  �  �  q  �  �  �  �  �  �  �  �  �  �  �  �  ^  "  �  �  W    �  �  �  �  �  o  Q  7  !    �  �  �  y  T  2    �  �  c    +  ~  �  �  �  |  _  ;    �  �  _    �  �  [  �  7  d  [  E  D  @  =  7  1  '    �  �  �  z  R    �  �  Y  �  U    �  �  �  �  �  �  �  o  X  A  *        ,    �  �  �  �       0  >  J  N  O  J  8    �  �  �  I     �  @  �  �   g  �  �  �  �    
    �  �  �  �  h  0  �  �  Z  �  �    }  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  x  m  b  W  U  R  N  G  @  =  >  >  <  9  3    �  �  �  w  J     �    �  �  	  	>  	N  	Q  	O  	K  	<  	  �  �  $  �  �  1  9  �     h  d  `  ]  Y  V  R  K  B  9  0  '      �  �  �  �  �  �  Y  Z  \  V  L  B  6  +      �  �  �  �  �  �  ^  6  	  �  �  �  �  �  �  �  �  ^  B  /    �  �  �  S    �    A  8  N  �  C  �  	  C  ]  S  +  �  x  �      �  `  �  e    .  �  u  e  V  G  ;  0  %       "  #         �  �  �  �  �  W  w  z  t  q  v  �  �  �  i  <    �  �  �  S  �    |    R  9  $    �  �  �  �  �  �  �  j  G  %    �  �  �  �  w  	f  	l  	g  	R  	1  �  �  �  V  !  �  �  �  Y  �  t  �  
  6  B  e  O  :  "    �  �  m  ,  �  �  3  �  �  )  �    '  �  p  �  �  t  6  �  �  d    �  �  A  �      �  *  �  6  �  4