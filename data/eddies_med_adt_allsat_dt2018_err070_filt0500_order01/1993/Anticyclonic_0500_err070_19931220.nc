CDF       
      obs    6   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?��E���      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M�?   max       P;��      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ����   max       =��#      �  \   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?��
=p�   max       @E��z�H     p   4   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�=p��
    max       @vr�\(��     p  (�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @'         max       @O            l  1   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @�w�          �  1�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �o   max       >H�9      �  2X   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�ِ   max       B+�?      �  30   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A��r   max       B,H[      �  4   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ?Rg�   max       C�u      �  4�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ?X��   max       C�td      �  5�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          v      �  6�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          /      �  7h   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          )      �  8@   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M�?   max       P x      �  9   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��i�B��   max       ?�dZ�2      �  9�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ����   max       =��m      �  :�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?��
=p�   max       @E��z�H     p  ;�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?��G�{    max       @vr�\(��     p  D   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @#         max       @L�           l  L�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @�?�          �  L�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         >�   max         >�      �  M�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�>�6z�   max       ?�dZ�2     �  N�            	                  	   2      	         u         	   1   	   *         o      A   &                  )      #                         
                  k            N���N���N0I�NG�[NbbM�?O7M�O�թO��HOs�O�9*O@@�N��N9�N��P�Nmm N7$�O:O���O!˫O�`�O��O7y�P;��P xP0?hP�O��bO�	�Ow�NC�N*��O�&9M�G�O:��O��&O/�nN���N���N5g�O��N���N���NW�N��O#ךN�vN�U^O�o�Ns�Nd�gNa.N�}�����t��t��o��o�o��o%@  ;D��;�o;��
<49X<u<�o<��
<�j<���<���<�/<�/<�`B<�`B<�`B<�`B<�h<�h<�h<�<��=o=o=C�=t�=t�=�P=�P=�w=�w=0 �=@�=D��=H�9=ix�=��=�hs=��=��P=���=�-=�E�=��=��=�h=��##/29;7/#����������������������������������������`aht����th``````````),+,))sqst����utssssssssss����������������������� ��
#/3:></#��%)5:BN[`gtwtdNA5942;<HU\ahjga`UNH<994116BO[ht�����tOB?74 "#+/<HU\dbaUSH</%# `[anz����{zzna``````qt��������|tqqqqqqqq��������������������`\\ct�������������t`�)),)$	������A?BGNNNVZZNBAAAAAAAA���������'6H[eillgb[OB6%�������������������#/<Hanz���znaU<#���������������������������������������������
 '--%"
����/UcgwytsynU/#
����)6BPRQE6)����//*,+3@[g�����t[IB5/�)5BFHGEB5)������������������������� #.<ADC<2�������

����������������}�����������������������		�����������������������������������������������������������������!)67>AA;62)"����������������������������������������������������������������$(0BKBA6)����������������������()69;986+)$unqxz}�����zuuuuuuuu"##,/2<@HHJHF</-&#""^\^abkmz|�����{zmfa^'&$&)*5<BGHDB65)''''`[UWYahnsz}~|zxrna``��������

�����)5BGBB5)PTV[gghtutpgb[PPPPPPypnvz�������}zyyyyyy��������������������D�EEEEEEED�D�D�D�D�D�D�D�D�D�D�D�������������������������������'�)�3�5�4�3�.�'�#���!�'�'�'�'�'�'�'�'�L�Y�\�a�^�Y�L�F�B�G�L�L�L�L�L�L�L�L�L�L�׾����������׾־о׾׾׾׾׾׾׾��n�zÇÉÓÇ�z�u�n�l�n�n�n�n�n�n�n�n�n�n�A�N�Z�g�s�~�|�s�o�d�Z�O�N�K�A�=�7�5�>�A�;�G�K�T�`�m�|�����~�y�`�.�'� ���"�*�;�H�a�m�q�v�z������z�l�d�a�R�H�;�;�7�C�H���������������������������������������������ʼ׼����ּż�����������������������������������������������������/�<�E�D�A�?�<�/�+�#�"�#�'�.�/�/�/�/�/�/����������������������<�H�U�W�U�L�H�<�9�5�<�<�<�<�<�<�<�<�<�<�6�B�O�[�g�p�r�o�h�O�6�������������6�Z�\�]�^�Z�Z�M�D�A�;�A�F�M�U�Z�Z�Z�Z�Z�Z�;�G�N�T�`�`�`�T�G�;�7�:�;�;�;�;�;�;�;�;�"�.�;�G�T�`�Y�T�G�;�.�"��	���	���"�ʾ׾���	��	�����׾ʾ������������������(�3�5�<�;�5�3�3�(�������������)�5�8�:�B�E�B�5�(��������������r�������������������r�f�[�Y�M�Y�f�m�r������&�(�(�������߽ݽнƽн׽���[�g��o�[�B�)���������������5�[�������������������z�a�]�_�|�����������ż'�4�6�7�@�;�$��'����ܻͻϻŻûܻ��'�����������$�6�9�+������������Ʈƪ���)�/�5�F�N�g�t�x�q�g�[�B�5�&������)�M������������u�s�c�Z�U�M�,�,�(�&�4�C�M�f�s����������������������������s�`�`�f�#�/�7�:�2�/�,�#���#�#�#�#�#�#�#�#�#�#�n�q�zÃÇÈÇÄ�z�v�n�i�a�a�a�l�n�n�n�n��'�3�B�S�_�d�d�^�Y�L�3�'������������������������������������������������ÓàìðùûùùìàÓÇ�|�z�q�o�q�zÇÓ��/�;�@�C�B�?�>�;�5�/�"��	������	������������	��
�	�������������������׼'�+�4�4�'�$���������������%�'E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�Eٿ.�"��	��	���"�.�4�.�.�.�.�.�.�.�.�.�!�:�S�c�~�{�g�S�:�!�	���׺����� ����!���������������������y�w�l�l�l�t�y�z�����������������������~�z�r�p�r�~�����������M�Y�f�r�u�r�l�f�Y�O�M�J�M�M�M�M�M�M�M�M�o�{�|ǈǋǈǂ�{�o�m�b�\�V�O�R�V�b�m�o�oŭŹ��������������������ŹŭŠşŜŠŦŭ�U�b�n�{ņŇŌŇŀ�{�n�b�_�U�S�O�U�U�U�U�����ûлܻ�����ܻлû�������������DoD{D�D�D�D�D�D�D�D�D�D�D�D{DyDnDjDfDiDo�5�5�=�<�;�5�(�(� ��(�3�5�5�5�5�5�5�5�5�A�N�Z�^�g�g�g�]�Z�T�N�G�A�9�A�A�A�A�A�A�����!�&�.�.�.�!�������������������EuE�E�E�E�E�E�E�E�E�E�E�E�E|EuErEuEuEuEu $ & r R j ` - . @ + & . d [ F # n F \ " $ Q P 9 L R Q 6 E 3 | T _ 3 s  Z " e & N l n @ ) 6 L 8 4 0 L V K 9  �  �  �  �  �  +  �  �  K  1  �  �  �  l  :  �  �  i  g  �  X    E  �  �  �  N  �  n    �  �  b  �    �  [  r  4  �  Q  d    �  l  �  �  �      �  �  �  �o;o�o:�o:�o;o<�o<�<���<D��=aG�=#�
<ě�<�t�<ě�>\)<�h<�=t�=���=�P=�\)=8Q�=T��>V=Y�=��=�C�=P�`=y�#=H�9='�=49X=��-=�w=�hs=T��=ix�=Y�=q��=L��=��T=�7L=��P=���=�{=�^5=�E�=�;d>H�9=ȴ9=��>+>n�B�DB��B ��B<_BmB	�B��B�}B��B�VB=�B�+B�!B�B�B
�"B�B/�B�Bd�B$�B�HB">�B"yiB7�B/B��B��B�	B��B�1B�B3fB��B�dB!�	B�vB
�B!W�Br�B"��B�B+�?Bk�BbCB�<A�ِB(4B��B�-B(B�&BhXBp�B��B��B �B@
BA�B	��B�BV�B�B8 B�BϽBlB�:B�	B
��B=�B=�B�B{�B��B@9B">TB"D&B@�B��B~�BشB��B�(B��B�BA�B�VB�B"?fB�JBDdB!/�BS�B"��B0B,H[B<�B@�Bw�A��rB?�B�B��B�B	:|B@QBV�C�M#?Rg�?�%�?���AVRAȺ�A�0,Agz�A��A���@���A�88A�g�A2��A�S@AׇWA=��Ae]-AaY�AQm�A���A�y�@�+�A0[�A�)�A�'�@�6aB:�A�kAA�AI�`A��A�,g?��7A��AʟA��-A�@��C�uA^J�@z�)A[�@�@ۯKB�6A�w�A�o�@���C��SA�)A���A
ؠC��C�IE?X��?�
?��FAU�A�QA�D]Ah��A�~�A�5�@� �A�p�AÃA3!A�B�A�WA>|Ad��Ab��AQ�A�nuA��@��XA1 A�vA�|�@��B��A��6AC�AJ��A��iAǂZ?�زA�ohAʃ�A�{�A��H@�1C�tdA^�@{��A@
tT@�h�B{�A�u�A�L�@�!C���A��PA�{A
��C��            
                  	   3      
         v         
   1   
   +         o      B   '               	   *      #                  !      
   	               l                                    !                        %            #      !         /   )   /   '      #                                    )                                                                                                                  )      !                                                                              N���N�XeN0I�N#K�NbbM�?Of�O-�OD*�Os�O���N�1RNN7�N9�N��OVYxNmm N7$�O:O7!7O!˫O��{O��O�Or��P xO��O��!O�N�O�->N�s�NC�N*��O��M�G�O��O��&O/�nN��|N���N5g�OY%�N���N���NW�N��O#ךN�vN�U^O�o�Ns�Nd�gNa.N�:   �  ]    �  �  �  �    3  8  �  �  �  ]  :  :  �  �  �  �  �  _  P  0  8  +  �  I  �  �  2  �  m  �  �  ~  /  !  W  �  �  �  �    L  �  �  �  �  )  �  \  P  �����o�t���`B��o�o:�o<#�
;�`B;�o<t�<�1<�C�<�o<��
=���<���<���<�/=H�9<�`B<��<�`B<��=�1<�h=H�9=t�=+=\)=��=C�=t�=<j=�P='�=�w=�w=49X=@�=D��=q��=ix�=��=�hs=��=��P=���=�-=�E�=��=��=�h=��m#/29;7/#����������������������������������������ddht����thdddddddddd),+,))sqst����utssssssssss��������������������
#*/15554/#
%"#),5ABNZ`fe[YNB5)%942;<HU\ahjga`UNH<99338BO[ht����{h[OB:63'),/<HMUWURHC<2/''''\anz����znca\\\\\\\\qt��������|tqqqqqqqq��������������������rnmqt�������������yr�)),)$	������A?BGNNNVZZNBAAAAAAAA���������.+*-6;BO[\`a_[ZOB76.�������������������#/<Hanz���znaU<*#������������������������������������������������

����/UcgwytsynU/#
����%)*)�����312;H[gt������wgNB53	 )5BDFDA5)	��������������������
#/499/$#����

����������������}������������������������������������������������������������������������������������������!)67>AA;62)"���������������������������������������������������������������!%%(������������������������()69;986+)$unqxz}�����zuuuuuuuu"##,/2<@HHJHF</-&#""^\^abkmz|�����{zmfa^'&$&)*5<BGHDB65)''''`[UWYahnsz}~|zxrna``��������

�����)5BGBB5)PTV[gghtutpgb[PPPPPPypnvz�������}zyyyyyy��������������������D�EEEEEEED�D�D�D�D�D�D�D�D�D�D�D���������������������������������'�)�3�5�4�3�.�'�#���!�'�'�'�'�'�'�'�'�L�Y�Z�`�]�Y�L�G�C�J�L�L�L�L�L�L�L�L�L�L�׾����������׾־о׾׾׾׾׾׾׾��n�zÇÉÓÇ�z�u�n�l�n�n�n�n�n�n�n�n�n�n�A�N�Z�g�s�v�u�s�k�g�`�Z�N�A�A�:�8�A�A�A�T�`�m�r�y�y�|�y�s�m�`�T�G�;�4�0�4�;�H�T�H�T�a�l�m�q�z�z���z�m�a�T�K�H�@�@�>�D�H�������������������������������������������ʼּ߼��޼Ҽ�����������������������������������������������������������<�C�B�?�<�<�5�/�&�)�/�1�<�<�<�<�<�<�<�<����������������������<�H�U�W�U�L�H�<�9�5�<�<�<�<�<�<�<�<�<�<�)�6�B�O�Q�X�[�W�O�I�B�6�)�!������)�Z�\�]�^�Z�Z�M�D�A�;�A�F�M�U�Z�Z�Z�Z�Z�Z�;�G�N�T�`�`�`�T�G�;�7�:�;�;�;�;�;�;�;�;�"�.�;�G�T�`�Y�T�G�;�.�"��	���	���"���ʾ׾�������׾ʾ������������������(�3�5�<�;�5�3�3�(��������������#�/�4�7�?�B�?�5�(����������	��r�������������������r�f�[�Y�M�Y�f�m�r�������#�%���������ݽҽ۽����5�B�N�[�b�g�j�j�c�[�N�B�5�)�����)�5�������������������z�a�]�_�|�����������ż�'�-�0�2�3�1�'�������ݻݻ������������$�1�4�/�'���������������������*�5�B�N�`�g�q�t�m�g�[�5�)������'�*�f������������p�f�Z�M�?�7�0�.�4�A�M�R�f�����������������������������������������#�/�7�:�2�/�,�#���#�#�#�#�#�#�#�#�#�#�n�q�zÃÇÈÇÄ�z�v�n�i�a�a�a�l�n�n�n�n�'�3�@�L�V�Y�\�X�L�E�@�3����������'����������������������������������������ÇÓàìíùùù÷ìáàÓÇÀ�z�t�r�zÇ��/�;�@�C�B�?�>�;�5�/�"��	������	������������	��
�	�������������������׼��$�"������������������E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�Eٿ.�"��	��	���"�.�4�.�.�.�.�.�.�.�.�.�:�S�_�s�m�_�S�F�:�-�!�������!�-�:���������������������y�w�l�l�l�t�y�z�����������������������~�z�r�p�r�~�����������M�Y�f�r�u�r�l�f�Y�O�M�J�M�M�M�M�M�M�M�M�o�{�|ǈǋǈǂ�{�o�m�b�\�V�O�R�V�b�m�o�oŭŹ��������������������ŹŭŠşŜŠŦŭ�U�b�n�{ņŇŌŇŀ�{�n�b�_�U�S�O�U�U�U�U�����ûлܻ�����ܻлû�������������DoD{D�D�D�D�D�D�D�D�D�D�D�D{DyDnDjDfDiDo�5�5�=�<�;�5�(�(� ��(�3�5�5�5�5�5�5�5�5�A�N�Z�^�g�g�g�]�Z�T�N�G�A�9�A�A�A�A�A�A�����!�&�.�.�.�!�������������������EuE�E�E�E�E�E�E�E�E�E�E�E�E~EuEuEuEuEuEu $ ' r I j ` ) ) 5 + ( $ g [ F  n F \  $ N P 2 5 R 9 ! E 0 @ T _ - s  Z " O & N U n @ ) 6 L 8 4 0 L V K <  �  �  �  X  �  +  B  r  �  1  j  �  �  l  :  �  �  i  g  y  X  �  E  F  �  �  �  �  ;  �  �  �  b      B  [  r  �  �  Q  �    �  l  �  �  �      �  �  �  �  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  �  �  �  p  \  F  .    �  �  �  �  V  $  �  �  Y  �  �  (  S  Y  ]  ]  [  Y  S  K  A  ;  :  6  0    
  �  �  �  �  f            �  �  �  �  �  �  �  u  X  8    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  ~  U  .    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  v  m  c  [  S  K  C  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �      #  �  �  �  �  �  �  �  �  �  �  �  y  Y  1     �  �  Y    �  �  �  �  �              �  �  �  [    �  �  .  �  �         (  /  3  3  .  $    �  �  �  �  t  F    �  �  �  8  6  5  2  .  *  %        �  �  �  �  �  �  �  �  �  v  �  �  �  �  �  �  o  U  2    �  W  �  k  ,  �  �    u  �    H  n  �  �  �  �  �  �  �  �  h  =    �  {  -  �  �  p  �  �  �  �  �  �    
                      j  �  ]  X  S  N  I  C  >  9  4  /  )  !      
     �   �   �   �  :  8  6  4  2  .  %      
     �  �  �  �  �  �  �  �  �  	8  
$  
�  �    ~  �    5  6    �  �    T  
}  	|  �  5  5  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  ~  {  x  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  {  �  z  s  v  {  u  k  _  T  A  )    �  �  �  �  �  �  �  �     I  j  �  �  �  �  �  �  �  �  �  �  h  *  �  :  �    ^  �  �  �  �  �  �  �  �  ~  e  G  *    �  �  �  �  �  L    >  ^  ]  W  L  B  G  V  Q  ;    �  �  ^    �  ,  �  %  v  P  ?  +      �  �  �  �  �  �  �  t  Z  <  0    �  �  T    #  /  .  &      �  �  �  q  @  
  �  �  \  "  �  �    	3  	�  
9  
~  
�  
�  
�    '  6  5  
  
�  
1  	�  	  7    ~  8  +                      #    �  �  �  �  6  �    �    a  �  �  �  �  �  �  �  x  ^  5  �  �  C  �    �        H  I  B  4    �  �  �  �  P    �  �  ;  �  �  A  y  �  �  �  �  �  �  �  �  �  z  Q  (    �  �  �  �  n    �  �  �  �  �  �  �  �  �  �  �  �  �  �  y  Z  0  �    �   �  1  #      �      )    �  �  �  Y  !  �  �  �  �  �  �  �  �  �  �  �  x  m  b  W  L  @  4  (          �  �  �  m  �  �  �  �  �  n  E    �  �  �    Z  4    �  �  �  �  4  h  �  �  �  �  �  �  f  2  �  �  Y    �  O  �  X  �    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �      #  2  A  b  t  ~  z  n  ^  K  9     �  �  �  D  �  �    �  �  1  |  /  &    
  �  �  �  �  �  �  �  k  I  "  �  �  �  }  m  j  !        	  �  �  �  �  �  �  �  �  n  L  '  �  �  �  �    :  U  A  -      �  �  �  �  �  \  0  �  w  	  �  K    �  t  g  Y  K  <  ,      �  �  �  �  �  �  �  l  W  <    �  �  �  �  �  �  �  �  �  �  {  q  g  ]  S  I  ?  5  +  !  �  �  �  �  �  �  �  �  n  B    �  �  z  =  �  �  {  :  
  �  �  �  }  s  g  \  P  C  3       �  �  �  �  �  b  H  .    �  �  �  �  �  �    b  <    �  �  {  F    �  �    E  L  =  /    
  �  �  �  �  �  �  ~  f  9    �  �  �  w  c  �  �  �  �  q  ^  J  2    �  �  �  �  {  U  /    �  �  �  �  �  ]  6    �  �  �  �  h  E  '    �  �  �  �  s  @    �  �  �  �  �  �  �  x  j  \  N  ?  /       �  �  �  �  �  �  �  b  (  �  �  �  O    �  �  :  �  �  Q  �  �  :  �  k  )  
  �  �  �  �  n  1  �  �  �  g  �  �    5  +  
�  �  C  �  �  �  �  �  �  s  `  M  ;  '      �  �  �  �  �  �  {  \  H  4    	  �  �  �  �  �  �  q  d  X  <    �  �  x  E  P  ?  0      �  �  �  p  "  �  �  E  �  �  [    �  `  
  �  �  �  �  �  z  d  G  (    �  �  �  j  =    �  �  '  L