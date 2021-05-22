CDF       
      obs    /   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?pbM���   max       ?�j~��"�      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M��   max       P�d�      �  h   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       �e`B   max       >��
      �  $   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?z�G�   max       @E��Q�     X  �   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       �ۅ�Q�    max       @vp��
=p     X  '8   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @         max       @P�           `  .�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�e        max       @��          �  .�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ���
   max       >�dZ      �  /�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�"�   max       B-�      �  0h   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�v   max       B-Qr      �  1$   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >�L�   max       C��      �  1�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >�5   max       C�	�      �  2�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max         ?      �  3X   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          C      �  4   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          C      �  4�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M��   max       P�xw      �  5�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?����l�   max       ?㴢3��      �  6H   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       �49X   max       >��
      �  7   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?=p��
>   max       @E��\)     X  7�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��(�`    max       @vj�Q�     X  ?   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @#         max       @P�           `  Fp   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�e        max       @�G           �  F�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         A   max         A      �  G�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�*�0��   max       ?�-V     �  HH         ?            W      	   	               !         	   	   $         1   D   A      c      	      6            ^         T      !   W         	         /PI�;N���P�'�NIN�z�N�ݗP�d�OI!N|��NdmlN�	�M��O"V9N�1�P*�%O&9xO��N��O	�%O��N��OQ�QP�O��yP�NmfOޒ#O��<O	�N�n�P��,N��O첵OJ�zP��O]��N�`�O���O��O��O���N�2�N��ENZD'N5��O,�N�/��e`B�49X�#�
�#�
�t��ě����
���
���
��o%   ;�o;�`B<D��<u<u<�1<�1<�1<�1<�j<ě�<�/<��=o=o=\)=t�=t�=�P=�w=<j=<j=L��=P�`=Y�=Y�=]/=e`B=e`B=���=��w=���=���=�^5>V>��
����0K^hf_I:-#	�����������������������!#/=N[��������t[L,!OORZ[_hjhd[OOOOOOOOO-*,//:<>HLQTSH</----���������������[outgN=7&���YXW[gt��������togc[Y����������������������������������������MMOO[fhjt|ztqh\[ZOMM��������������������ffgit����������tnjgfwmnvz����������zwwww������
#<KQM</#����4127<HRUanunkbaUTH<47BQ\t��������tg^XOB7��������������������TPKJTanoqrrrnlea]YUT�#0<IYaa]UOI0"�����������������	
#)-122/#
��������������������������������ru����"/4(�������r����������������������������������������sps|��������������zs����������������������)20)���������)5?KMJ) �����MKMN[dgt����|tga[ZNM�������� �����������%&)5BO[_a`\[SONB4+)%�����6BJPF@6.)����������� ��������������������������$%)/8;HTamsvvrmaTH/$"#/7@HPY`ZUH</('"��������� ���������������
),(#
��tttv�������������tttMKLOT[ehomhf][WOMMMMVRW[ehmsnhe[VVVVVVVV+./<HILH</++++++++++H<;8644457<DHLPRPMJH//<@A<:/##/////�������û������������x�g�F�-�!�&�:�S�x��ĳĿ������ĿĳĨĦĞĦĭĳĳĳĳĳĳĳĳ��6�O�tčĚĞĚč�t�[�6��������������y�z�������������~�y�r�x�y�y�y�y�y�y�y�yD�EEEEE*E,E*E#EEED�D�D�D�D�D�D�D������������ĿƿĿ������������������������0�I�nŎşŏŉ�n�U�0������Ŀĳĸ�����0�����������������������������������������'�3�@�F�J�@�@�3�)�'�$�$�'�'�'�'�'�'�'�'�[�]�g�o�g�_�[�P�N�B�5�5�5�B�N�X�[�[�[�[������������������������������<�<�>�<�<�:�/�$�#�#�#�#�/�:�<�<�<�<�<�<�(�5�=�A�I�I�A�<�5�,�(����
�
���$�(ƧƳ��������������ƳƬƧƚƓƚƦƧƧƧƧ��(�5�I�Q�Y�S�A�6�8�5������߿������������������������������������T�_�y�������������y�`�\�T�J�K�T�T�G�M�T���������Ŀ˿οĿ������������������������g�s�����������������s�g�]�Z�N�I�N�Z�e�g����������������������o�f�`�T�Y�r������A�N�R�D�B�C�A�7�5�(�#�����"�(�5�@�A�`�m�������������y�m�`�T�G�D�?�D�E�G�T�`��������%�&�������ïíòïù�����뻞�����ûѻ׻ۻ�ܻл������u�l�r�{������������;�H�K�B�"�	���������j�p�����������A�B�N�V�Z�c�Z�N�E�A�@�A�A�A�A�A�A�A�A�A���Ϲ�����$������Ϲ��������������������������������������������w�t���������A�M�S�V�Q�O�M�H�E�A�4�2�2�(�!�!�'�(�4�A�f�g�k�s�t�s�r�o�f�Z�W�V�Y�Z�e�f�f�f�f�f¿������������t�g�;�.�B�[§²¿�/�;�H�I�J�H�E�D�>�;�/�&�"��$�"�"�"�-�/�	���"�*�5�5�/�"�������������������	�׾��������׾ʾ����������������ʾϾ׺e�r�������������ĺ������~�b�Y�B�K�G�N�e����*�6�:�C�D�M�C�2�������������4�A�F�M�V�U�O�M�A�<�4�(�&�&�(�-�4�4�4�4Ŀ���������"�!����������������ļĶĿ�M�Z�f�s�|�z�s�f�U�A�4�(������(�4�M�G�S�`�l�y�����������y�l�`�S�L�J�G�G�@�GEuE�E�E�E�E�E�E�E�E�E�E�E�E�EuEpElEmEuEu������������������������������������������'�4�;�@�K�@�?�4�'����
�������Y�f�r�x�����r�f�Z�Y�P�Y�Y�Y�Y�Y�Y�Y�Y²¿����¿¿²®¦¦²²²²²²²²²²ŠũŭŹ�����������������������ŹŭŠǭǭǧǡǔǉǈǃǄǈǉǔǡǣǪǭǭǭǭǭ 9 .  ; + " 0 2 @ U B ^ ) 4 M @ \ 3 T 1 Z 8 $ * 6 q F  h � Y V ` ; 9 / ! . e \ G k 1 E 6 | 8    |  ~  �  -      �  �  �  �  �  Q  W  �    j  �  �  L  �  =  �  �  �  2  f    �  {  �  O    �  �  �  �  �  �  K  �  �    �  s  Z  �  �<�t����
>��-���
<�t�:�o=��T<49X;�o;��
;ě�<o<u<�t�=D��=��=�P<�<�h=e`B<�/=@�=���=ȴ9=Ƨ�=�P>1'=�%=8Q�=49X=�v�=T��=�hs=�7L>z�=��=�\)>I�=�C�=�9X>%�T=�E�=\=�-=�
=> Ĝ>�dZB$��B��B	;�B:�B�oB�B�*B	��B!�8B��BxB��B
9�B !�B�OB2wB	w�BYB5 B&QB��B6�BH�B��B�BS�B��B�B!F�BֳB��B	@*B@BZ�B�B=�B��A�"�B�VB-�B��BR�BMB�B��B��B �B$c�B��B	?�BCjB�bBABJB	��B!�/B�
B�rB��B
�B A�B��B?�B	��B�MB�B&'�B��B@BH�B��B��B�B�B?B �*B�B��B	��B��BA B�:B;�B4A�vB��B-QrB�B)�B@XB��B��B�`B3�@��)A�֜A�ޡAi-C�_cAtaTA�հA��S?�ڦA��yA1�Aµ�A�c�B`~A���AһFAlYAu��A���@�2A�|gAi��A�s6@��A��A�|->�L�A��A:�aA@!�A�rA�A��aARD@�A��A:�(A��A;��A��C��@��0@�O�@�b�A�� A��BR�@��A�~�A�q�A�xC�X>As�A�E�A���?��hA�q�A0��A�A��_B�A�|WAҀ?AmVAv� A�{�@�8�A���AiŧA�\�@�̰A��A�}�>�5A�r�A:cWAA/�A� �A���A�}�AR�+@&�A�a�A;A�OA7GA �C�	�@�)m@��@�:\A���A��>BC�   !     ?            W      
   
         	      "         
   	   $         2   D   B      c      
      7            _         T      !   W         	         /   1      7            C                        )      !         #         #   !   C      #            ;      )      )               !   !                     /                  #                        '                              C                  3      )                                          P(��N���O�,�NIN�z�N�ݗOϠ�OI!N!��NdmlNfi�M��O��N�1�O�6�N3�'NV�BN��N�ۈO�SN��OA�O��ZOi�@P�xwNmfOx��O���O	�N�n�PDd�N��O�^�O"cAO�\�O]��N���OJ��O��O�'�ON}�N�&�NM�7N1�N5��NN�/�  a      �  Z  @  1    k  �  r  (  @  9  a  �  �  5  g  �  �  �    	M  �  �  �  b  t  �  �  ~  |  w  
X  w  �  �  �  [  m  C  v  �  �  �  
�#�
�49X>��#�
�t��ě�=8Q컣�
�D����o:�o;�o<o<D��<�1<���<�h<�1<�9X<�h<�j<���=t�=H�9=+=o=��=�w=t�=�P=H�9=<j=@�=T��=���=Y�=]/=���=e`B=ix�=��=���=���=��
=�^5>\)>��
����#0U`^UI<7#������������������������@DIN[gt�������tg[NB@OORZ[_hjhd[OOOOOOOOO-*,//:<>HLQTSH</----���������������#(,(" ��YXW[gt��������togc[Y����������������������������������������NOPS[hstwtlhb[RONNNN��������������������plhhmt�����������tppwmnvz����������zwwww����/<BJI</#
�����89<HOU[UKHC<88888888Y[^gkt}���utg[YYYYYY��������������������QLLU[ahnopqpnhba\WUQ#0<INX[ZVOID0#����������������
#(,/111/#
�����������������������������	�������tv����!/4(�������t����������������������������������������wtw{��������������zw����������������������)20)���������)5@GHC5	 ���MKMN[dgt����|tga[ZNM��������������������()46BOY[^_^[WOFB7.)(�������*260)������������ ��������������������������6459;@HTailmgaWTH@;6"#/7@HPY`ZUH</('"����������������������
!#
���xuw���������������xxNOO[[\hkjhb[RONNNNNNXSX[ghhrmha[XXXXXXXX+./<HILH</++++++++++;9654467<CHLOQPLHH<;//<@A<:/##/////�������������������y�q�W�F�*�-�:�S�_�x��ĳĿ������ĿĳĨĦĞĦĭĳĳĳĳĳĳĳĳ�B�[�h�o�x�z�v�n�h�[�O�B�6�"����!�6�B�y�z�������������~�y�r�x�y�y�y�y�y�y�y�yD�EEEEE*E,E*E#EEED�D�D�D�D�D�D�D������������ĿƿĿ�������������������������#�0�<�I�^�`�\�U�I�0�#�
������������������������������������������������������'�3�?�@�E�@�8�3�0�'�&�&�'�'�'�'�'�'�'�'�[�]�g�o�g�_�[�P�N�B�5�5�5�B�N�X�[�[�[�[���������������������������������<�<�>�<�<�:�/�$�#�#�#�#�/�:�<�<�<�<�<�<���(�5�A�F�G�A�7�5�(�&��������ƧƳ��������������ƳƬƧƚƓƚƦƧƧƧƧ�(�?�H�L�A�4�.�.�(������������(�����
��������������������������������m�q�y�����������y�v�m�h�i�k�m�m�m�m�m�m���������Ŀ˿οĿ������������������������s�������������������s�g�`�Z�N�N�Z�g�i�s�����������������������r�j�f�Z�d�r�{��A�N�R�D�B�C�A�7�5�(�#�����"�(�5�@�A�`�m�������������y�m�d�`�T�G�E�@�E�G�T�`����
��������������õôùú�����뻞�����ûǻλллɻû������������~������������;�G�K�A�"�	���������m�s�����������A�B�N�V�Z�c�Z�N�E�A�@�A�A�A�A�A�A�A�A�A�ùϹܹ�����
�����ܹϹ������������������������������������������������������A�M�S�V�Q�O�M�H�E�A�4�2�2�(�!�!�'�(�4�A�f�g�k�s�t�s�r�o�f�Z�W�V�Y�Z�e�f�f�f�f�f¿����������²�t�g�T�A�E�[¥²¿�/�;�H�I�J�H�E�D�>�;�/�&�"��$�"�"�"�-�/�	��"�)�0�4�5�/�"�������������������	�׾�������޾׾ʾ��������������ʾվ׺e�r�~�����������������~�r�i�a�X�S�U�Y�e����*�6�:�C�D�M�C�2�������������A�D�M�U�U�N�M�A�@�4�(�'�'�(�4�6�A�A�A�A����������
���	���������������������ؾM�Z�f�s�|�z�s�f�U�A�4�(������(�4�M�G�S�U�`�y�����������y�l�`�S�M�J�H�H�A�GE�E�E�E�E�E�E�E�E�E�E�E�E�EuEtEsEuEyE�E������������������������������������������'�2�4�@�A�@�9�4�'�#����#�'�'�'�'�'�'�Y�f�r�w����r�f�\�Y�T�Y�Y�Y�Y�Y�Y�Y�Y²¿����¿¿²®¦¦²²²²²²²²²²ŭŹ�����������������������ŹŭūūŭǭǭǧǡǔǉǈǃǄǈǉǔǡǣǪǭǭǭǭǭ ; .  ; + " S 2 ? U M ^ ) 4 V 4 [ 3 K # Z :    5 q 7  h � ] V _ /   /   e \ 2 m + C 6 w 8    �  ~  �  -        �  C  �  �  Q  -  �  �  M  �  �    �  =  �  �  �  !  f  �  f  {  �  ~    }  c    �  �  �  K  �  �  �  V  Y  Z  v  �  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  T  Z  _  `  X  N  C  5      �  �  �  z  r  m  g  3  �  �    �  �  �  �  �  �  �  �  �  �  �  �  h  <    �  �  `     �  �  �  �    P  e  (  �  �    �  f  �  �  e  k  	  �  �  �  x  j  \  L  4      �  �  �  �  �  �  m  W  ?  &     �  Z  6    �  �  �  l  <    �  �  J  �  �    �  �  �  �   �  @  <  8  3  ,  $        �  �  �  �  �  �  x  e  f  o  x    I  Z  f  x  �  �  g  R  �    1    �  J  �  �  c  �  ?           �  �  �  �  �  �  p  U  =  &  (    �  �  K    U  ^  g  i  k  b  V  B  *    �  �  �  �  �  f  N  =  ]  }  �  �  �  �  �  o  ]  M  =  .        �  �  �  �  �  �  v  o  p  q  r  p  n  k  f  `  Y  P  C  7  (     �   �   �   �   �  (  1  ;  D  N  R  F  ;  /  #    	  �  �  �  �  �  �  �  �  4  9  >  =  9  4  +  #      �  �  �  �  �  v  c  S  F  9  9  1  (             �  �  �  �  �  �  �  �  �  �  �  �    7  U  `  `  Y  G  B  =  "  �  �  �  >  �  �  �  �  #  �    2  @  O  ^  n  �  �  �  �  �  �  b    �  "  �  �  8   �  K  ?  :  E  R  T  U  j  s    �  �  �  `  ;    �  �  �  w  5  3  1  .  +  #         �  �  �  �  �  �  o  S  :  ;  <  e  f  g  d  ^  X  O  G  9  *      �  �  �  �  �  �  �  �  u  �  �  �  �  �  �  �  �  f  >    �  �  �  _  &  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  m  I  %     �  �  �  �  �  �  �  �  �  �  �  �  k  N  '  �  �  r  -  �  �  �  �  �       �  �  �  �  �  �  �  c  9  �  �  D  �  R  �  �  	  	6  	@  	E  	M  	?  	%  	  �  �  �  E  �  |  �  Z  �  �  [  �  �  ~  U  .    6  '    �  �  u  0  �  k    �  �     �  �  �  �  �  �  �  �  �  �  �  v  ^  F  .    �  �  �  �  �  
�  �  �  D  z  �  �  �  o  2  �  x  
  
�  	�  	>  a  H  �  �  P  Z  a  _  X  M  9    �  �  �  _  (  �  �  �  }  B  �  �  t  m  f  Z  N  F  @  ?  A  :  +      �  �  �  �  k  &   �  �  �  �  �  �  �  u  c  Q  >  *    �  �  �  �  �  `  ?    ~  �  �  �  �  �  �  �  C  �  �    �  Y  y  W  �  �  8    ~  t  k  b  X  O  E  ?  :  5  ,        �  �  �  �  �  �  g  {  w  n  ]  D  #  �  �  �  _  $  �  �  �  S    �  �  �  Z  i  u  v  r  k  _  O  <  '    �  �  �  e    �  r   �   v  �  	#  	�  
   
,  
W  
C  
  	�  	B  �  g  �  ~  $  �  g  �    �  w  g  T  :    �  �  �  �  e  9    �  �  d    �  �  m    �  �  �  ~  d  H  ,    �  �  �  �  r  N  '     �  �  �  �  
[    �    i  �  �  �  �  �  �  7  �  K  
�  	�  �  �  �  7  �  �  �  �  v  d  P  7    �  �  �  �  a  ;    �  �  �    Z  X  N  D  8      �  �  �  �  I  �  �    �    �  �    �  R  �    9  Z  m  h  C  
  �  5  �  �  &  
  �  �  �  �  >  A  <  #  �  �  �  l  9    �  �  �  k  J  '    �  �  �  A  X  g  q  u  s  g  W  C  #  �  �  �  B  �  �  P  �  �  8  �  �  �  �  �  �  �  �  �  y  j  Z  J  5       �    )  P  �  �  �  �  �  �  �  �    r  d  V  F  4    �  �  }  R  *  �  �  �  }  k  Y  C  8  :  <  7  -  "      (  ^  �  �  G  
  �  ?  �  �  '  �  P  �  Y  �  ;  �    
x  	�  	7  �  �  �