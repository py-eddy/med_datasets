CDF       
      obs    1   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?�Z�1'      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       N��   max       P�1�      �  p   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       �ě�   max       =���      �  4   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?Q��R   max       @Eu\(�     �  �   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�
=p��    max       @vl(�\     �  '�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @)         max       @P            d  /H   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�@        max       @��          �  /�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       <T��   max       >s�F      �  0p   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�{|   max       B*(�      �  14   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A��   max       B)�)      �  1�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ?=54   max       C�J      �  2�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ?O��   max       C�I      �  3�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �      �  4D   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          C      �  5   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          ;      �  5�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       N��   max       P��      �  6�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?���@��   max       ?ܼj~��#      �  7T   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       �o   max       >�      �  8   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?Q��R   max       @Ek��Q�     �  8�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�\(�    max       @vl(�\     �  @�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @)         max       @P            d  H,   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�@        max       @�*�          �  H�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         C   max         C      �  IT   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�0��(�   max       ?ܹ#��w�     P  J            B               %                  x      	         #                              	   (   "                  	   
      .   �   o   9      0      M   Oø�N� �O���P�1�N��N��N��N��%O��ON��,O	�O��N��mN�JTPifN��N�߂O�r!O~�O�<�O,ԟO��O*��O�jN}�N�<iN��TO5IN/ˣN���Ox��P'Ou��O��O��N���N�H/N5N7�N*\xOo��PR��O��]O��AO��OiMO2��O�AhO@�O�ě�;ě�<o<o<t�<T��<u<���<���<���<��
<�1<�1<�j<���<���<���<���<���<�/<�/<�/<�h<�<��=o=+=C�=�w=#�
=0 �=8Q�=<j=@�=H�9=L��=L��=]/=]/=�o=�7L=��
=��
=��T=��=�1=�9X=��=��������
#(5<AH@</#!������������������������������������������[gh^]abN)����NLLO[]fhkiha[XPONNNN)5::5)!"()))*--)#<<AGHTalmnmidaTH<<<<���
#/7?EF=/#
�������������������������		
#%0;<910#
		kilt�������������|wk��������������������������������������)5N[_[QEBB5��@BOO[hlnth[ONB@@@@@@���������������������������!)���������	
��������� /<IPPLLOH?#
��-+*,/<HPU_algaUH<8/-����������������������������������������NOP[ht~�����xth][SON��
!
 �����������?EEHU[aba_VUMH??????snqz�����zssssssssss������
#%%#
������
�����������������������������KQ[ht��������th[TPOKKHIO[t����������tjRKORY[bgt���������tg[Oy{����������������|y � 	)7:;;83)>?BFN[`_b[NB>>>>>>>>c_^_gt�����|tmgcccc.,*/<G>?</..........����������������������	

�����������������  ��������zpqz������&' ���z��������
�����������!"���������������������������������������HHE</*#!#/<@EH������

������������""
�������/�;�T�a�d�a�`�U�H�/���	������������/���������������޽�������Ź����������*�2�6�+��������ŹūŢŰŹƚƳ�����#�2�(���Ƴƚ�h�J�@�D�=�O�\Ɖƚ�r�����������|�r�f�a�Y�U�Y�f�k�r�r�r�r������������|�y������������������������������	�	�	�����׾о׾���������"�/�;�<�C�B�;�7�/�-�"���������n�y���������y�m�`�T�G�@�<�>�D�F�T�`�k�nù��������������ùìà×àâìóùùùù�ּ��������������ּʼǼƼʼ̼Ѽּ־������������������s�Z�M�A�5�9�M�Z�s�����"�.�;�G�N�N�G�;�.�"��������������
�����������ܹԹֹܹ�������#�I�U�a�h�qŉŉ�{�b�U�<���Ļ�������H�J�T�U�[�Z�U�H�<�3�8�;�<�F�H�H�H�H�H�H�ѿݿ������������ݿٿѿȿοѿѿѿѿy�������������������y�e�`�G�9�7�=�G�`�y�)�5�B�J�P�D�5�)�������������������)�(�5�Z�g�����������s�Z�A��������(�����������������������������������)�)���������������������������ݿ�������������ݿԿѿĿ��ĿԿݻ-�1�:�G�O�M�F�@�:�-�!����	���!�*�-������������z�w��������������������������)�2�.�)���������������[�h�t�z�|�t�h�[�S�P�[�[�[�[�[�[�[�[�[�[�#�/�<�=�H�J�S�H�@�<�4�/�#�������#�����������������������������������������`�m�y�����������y�m�m�k�`�Y�Z�\�`�`�`�`�������������������s�f�W�N�P�X�f�s�x�����"�/�;�=�9�/��	�����������������������5�<�N�S�V�\�d�e�Z�N�A�5�(�!���	������ͽսݽ����ؽĽ����������y�r�x�����Y�e�r�~�����������������~�r�e�M�E�J�R�Y�tāĉčĒđčā�t�l�m�l�t�t�t�t�t�t�t�t�6�B�O�[�c�d�[�Z�O�B�6�,�)�#�)�.�6�6�6�6¦²¹¼²¦���ûлڻӻлû��������������������������s�u�������������s�f�a�f�o�s�s�s�s�s�s�����������������������������~�������������ʼּ�����������ּʼ�������������D�D�D�D�D�D�D�D�D�D�D�D�D�DzDpDrDwD{D�D������	��/�5�6�+�"�	���������������������g�t�t�g�d�[�Y�R�N�M�N�[�g����������������������ùìæàÞÔàù���o�b�V�N�N�V�^�b�o�{ǈǔǗǡǣǞǐǈ�{�oE�E�E�E�E�E�E�E�E�E�E�E�EuEnEgEbElE�E�E��n�{ŇŊŏŌŇŃ�{�n�b�U�O�O�T�U�Z�b�g�n a " y D T S � 1 : P @ : ] ! Y \ 6 � _ D 2 2 I  m + ) ( X 6 , f c 1 = 1 + K D \ 0 f : W ? 3 J 7 ?    @  �    �  �  A  �  (  3  �  4  J  �    V  �  �  �  <  \  v  _  �  E  o  �  |  {  p  �  �  �  -  p    �    T  @  \  �  #  B  �  m  �  �  �  �<���<T��<�=��<�j<u<�C�=C�=e`B=�P=o=,1<���=�w>t�<��=C�=<j=P�`=y�#=ix�=aG�=8Q�=L��=\)=,1=#�
=Y�=0 �=H�9=��=��w=u=�hs=��-=]/=�C�=}�=�%=�7L=�l�>s�F>C��>I�=��`>+=�h>.{>	7LB�ZB"��B��B-B>�BѐB��A�{|B��B!�{B%�BQ�Bv^B��B �B�B?�B8ZB��B�B[BBJBr�BڋB$R�B�`A��WB�OB�B!N�B6ZB	��B	�sB*(�B�_BN�B	��B��B!B�NB�pB��B��B��B�B��B6�B��BgsB��B"�<B�xBlB@BB�B��A��B�!B!�B%@
B?�B0�B 9B��B�DBA$B=�B�B9�B�B��B�_B�3B$O�B��B <�B�uBB�B!>,BC�B	�fB	��B)�)B��B;�B	��B��B!�B�B�*B�B��B�BJ�B�IB?yB=,B�,A�o�A0Q�A�9B�@�XAFV�AV�A���Ai{�A͙�AOLAD�A`�?=54A�"�A�RRA~�|Al�iA���A�`�AҫvAҁ A��@t��@�DA�9�A�y.A��AK��Ak�lADh�A��A��0A"�I@g3A�1A��YA���@��~AC��A�	�@�>�C��rA��yA�$8A�bB+�C�JA��A���A09A���B.@��AF�AT*�A��AiMA��
A��AF�sA`{?O��A�D�AāmA4LAi�A��\A���AҀ�A�A��J@w�@��A�h�A�}�A�~GAK&Al��AD�A�x�A��BA!�J@`A݋�A�c�A��:@���AB�7A���@�4C���A��UA�~�A�~;B�C�IA�~�            B               &                  x      	         #                              
   (   "                  	   
      /   �   p   9      1      M      #      %   C                        %         9         #      %                                    -      !                        ,      +                           ;                        %                  #      #                                    %                                    )               O��CN� �N�!�P��N��N��N��N��%OC�NB.VN��O��N��mN��TOĪ�N��N�߂O�r!O"�O��N�O��O
c�O
��N}�N�<iN��TO<bN/ˣN���OT��O���Ou��OC�O��N���N�H/N5N7�N*\xOo��O��gOa�O��'O��O,�HO2��O�AhO@�O  R     �  g  *  �  �  #    =  k  &  `  �  �  �  �    S  x    |  9  l    �  w      �  �  �  �  F    s  �  �  [  �  
L  k  �  	2  1  
1  �  �  	9�o;ě�<�C�<��
<t�<T��<u<���<�/<ě�<�j<�1<�1<�/=�hs<���<���<���=o<�`B=��<�/<��<��<��=o=+=��=�w=#�
=@�=L��=<j=]/=H�9=L��=L��=]/=]/=�o=�7L>�=��=���=��=�j=�9X=��=�������
#&2<C</#���������������������������������������������)5N[`YW[ZNB)��NLLO[]fhkiha[XPONNNN)5::5)!"()))*--)#<<AGHTalmnmidaTH<<<<���
#/3<A@<3/#
����������������������	
#/0840#
						kilt�������������|wk���������������������������������������)044.)��@BOO[hlnth[ONB@@@@@@���������������������������!)������������ ��������#/<HNOLLLH<#
4028<CHQUYUUH<444444����������������������������������������UOQ[[ht|�����vth_[UU��
!
 �����������?EEHU[aba_VUMH??????snqz�����zssssssssss������
###
������
�����������������������������WSQ[lt���������th^[WNLLO[t����������tgXNORY[bgt���������tg[O�������������������� � 	)7:;;83)>?BFN[`_b[NB>>>>>>>>c_^_gt�����|tmgcccc.,*/<G>?</..........����������������������	

�����������������  �����������������������������������



���������""����������������������������������������HHE</*#!#/<@EH������

������������""
�������/�;�H�N�T�W�Z�O�H�/��	������������"�/���������������޽�����������������������������ŹŵŰŹ����������ƚƳ���������%�$���Ƴƚ�h�W�Q�N�e�uƚ�r�����������|�r�f�a�Y�U�Y�f�k�r�r�r�r������������|�y������������������������������	�	�	�����׾о׾���������"�/�;�<�C�B�;�7�/�-�"���������`�m�y�����|�y�k�`�T�N�G�F�B�F�G�O�`�`ìù������������ùìàÞàëìììììì�����������ּϼ̼Ѽּ�������ﾌ�����������������s�Z�M�A�5�9�M�Z�s�����"�.�;�G�N�N�G�;�.�"������������������������������������
��#�2�<�E�L�M�H�<�0�����������������H�J�T�U�[�Z�U�H�<�3�8�;�<�F�H�H�H�H�H�H�ѿݿ������������ݿٿѿȿοѿѿѿѿy�������������������y�e�`�G�9�7�=�G�`�y���)�5�B�D�J�B�<�5�)���������������(�5�Z�m�����������s�Z�A�(��������������������������������������������)�)������������������������������
�����������ݿ׿ѿſѿݿ���!�-�:�F�F�N�L�F�>�:�-�!�������!�!������������z�w��������������������������)�2�.�)���������������[�h�t�z�|�t�h�[�S�P�[�[�[�[�[�[�[�[�[�[��#�/�7�<�D�H�L�H�=�<�/�#�"�����������������������������������������������`�m�y�����������y�m�m�k�`�Y�Z�\�`�`�`�`�f�s�����������������s�n�f�[�R�S�Z�]�f������"�4�8�2�/��	�����������������������5�<�N�S�V�\�d�e�Z�N�A�5�(�!���	������������Ľ˽ӽѽĽ��������������~�����Y�e�r�~�����������������~�r�e�M�E�J�R�Y�tāĉčĒđčā�t�l�m�l�t�t�t�t�t�t�t�t�6�B�O�[�c�d�[�Z�O�B�6�,�)�#�)�.�6�6�6�6¦²¹¼²¦���ûлڻӻлû��������������������������s�u�������������s�f�a�f�o�s�s�s�s�s�s�����������������������������~�������������ʼּ�����ۼּʼ�����������������D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D{D{D|D�D����	��.�4�5�)��	�����������������������g�t�t�g�d�[�Y�R�N�M�N�[�gìù��������������������������ùìçêì�o�b�V�N�N�V�^�b�o�{ǈǔǗǡǣǞǐǈ�{�oE�E�E�E�E�E�E�E�E�E�E�E�EuEnEgEbElE�E�E��n�{ŇŊŏŌŇŃ�{�n�b�U�O�O�T�U�Z�b�g�n Y " < E T S � 1 : a F : ] - 5 \ 6 � D E & 2 =  m + ) ( X 6 . ^ c " = 1 + K D \ 0 M * T ? & J 7 ?    �  �  �  �  �  A  �  (  �  a  �  J  �  �  �  �  �  �    D  �  _  ;  (  o  �  |  6  p  �  �    -  �    �    T  @  \  �  N  H  �  m  m  �  �  �  C  C  C  C  C  C  C  C  C  C  C  C  C  C  C  C  C  C  C  C  C  C  C  C  C  C  C  C  C  C  C  C  C  C  C  C  C  C  C  C  C  C  C  C  C  C  C  C  C  "  ;  M  R  P  F  7  #    �  �  �  �  �  p  Q    �  �  �     �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  r  ]  G  1  �  �  �  �  }  /  o  �  �  �  x  _  9    �  J  �    �   �  �  <  V  d  f  W  =    �  �  l  P    �  �  2  �  <  n  �  *  %  #  "         �  �  �  e  5    �  �  R    �  _  
  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �       �  �  �  �  �  �  �  �  �  �  �  v  b  M  8  #    �  �  �  #  "        �  �  �  �  �  �  �  i  '  �  �  K  �  �  P  S  �  �  �      �  �  �  t  2  �  �  6  �  a  �  G  �  �  �  �      0  :  A  A  ?  <  8  4  1  .  3  F  )  �  �  o  9  O  _  f  i  j  i  e  `  Y  S  M  B  1    �  �  |     �  &  #            �  �  �  �  �  �  m  C    �  �     �  `  \  X  T  N  D  9  .  !    �  �  �  �  �  �  r  9   �   �  y  �  �  �  �  �  �  �  �  �  �  s  [  G  7  #      7  \  	�  
   
	  -  {  �  �  �  �  �  �  x    
�  	�  	/  G    :  Z  �  �  �  �  �  �  �  �  �  �  �  x  l  `  S  F  /  �  �  Q  �  �  �  �  �  �  �  �  z  i  V  A  *  	  �  �  �  ^  (   �          
  �  �  �  �  �  �  �  �  �  �  Z    �  @  �  �    .  E  P  Q  H  <  +    �  �  �  Y     �  �  ?  �   �  m  t  d  R  :      �  �  �  �  j  L  "  �  �  7  �  �  �  �  �  �  �              �  �  �  z  7  �  �  _    1  |  q  T  +  �  �  �  u  K    �  �  %  :    �  �  T    �  0  6  8  5  -       �  �  �  �  �  y  W  2    �  �  d  /  k  l  k  g  e  _  Y  P  D  3      �  �  �  �  ]  +  �  �      	            ,  :  @  ?  =  <  :  +      �  �  �  �  �  �  �  x  g  S  ;    �  �  �  �  n  N  �  [   �   �  w  i  \  N  @  2  #      �  �  �  �  �  �  �  y  y  �  �  �             �  �  �  �  �  c  +  �  �  d  !  �  �  q              �  �  �  �  �  �  ^  8     �   �   �   �   {  �  �  �  �  �  |  r  e  W  A  &      �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  r  >    �  u    �  (  �  �  p  H  �  �  �  �  �  w  I    �  �    F  4  ?  �  �  0  �  �  �  �  �  �  �  l  L  +  
  �  �  �  �  {  ]  B     �  �  �  6  9  9  ?  A  F  E  >  :  5  '    �  �  �  �  _     �  �    �  �  �  �  �  o  I    �  �  I  �  �  �  K  �  m  �  T  s  h  ]  R  G  =  6  .  '         	    �  �  �  �  �  �  �  �  ~  y  q  e  R  8    �  �  �  c  &  �  �  %  �  K  �  �  �  �  �  i  Q  <  (  N  }  �  z  h  T  ?  +      �  �  [  T  M  F  ?  ,    �  �  �  �  l  B    �  �  �  j  `  V  �  �  �  �  �  ~  s  f  W  I  :  +      �  �  �  �  �  �  
L  
:  
,  
(  
  
  	�  	�  	�  	W  	  �  X  �  `  �  j  �    �  �  h  �  �      0  O  g  i  P    �  &  |  �  h  
�  �  z  v  �  �  �  K  t  �  }  Y     �  h  �     ?  @    
`  �  �  	  	0  	  �  �  x  4    �  k  #  �  i  �  h  �  �  `  v  )  1  #  "      �  �  �  �  �  ]  3    �  �  u  "  �  3  �  	�  
  
,  
1  
$  
  	�  	�  	|  	B  	  �  c  �  |  �    �  �   �  �  �  {  R  )  �  �  �  Y    �  �  X    �  N  �  :  �    �  7  ;  \  h  j  `  P  .  �  �  /  �  
�  
  	  �  �  �  U  	9  	  �  �  k  5  �  �  |  2  �  �  @  �  �    g  �  �  �