CDF       
      obs    7   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?Ǯz�G�      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M�5�   max       P�a%      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ���   max       >V      �  d   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�p��
>   max       @E�\(�     �   @   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�         max       @vffffff     �  (�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @0         max       @M�           p  1p   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @��`          �  1�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �u   max       >��^      �  2�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�8�   max       B-8      �  3�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A��m   max       B,��      �  4t   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ?��X   max       C�b      �  5P   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ?��I   max       C��      �  6,   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �      �  7   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          Q      �  7�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          -      �  8�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M�5�   max       P�      �  9�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?���rGF   max       ?��f�A�      �  :x   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��h   max       >�      �  ;T   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>������   max       @E��Q�     �  <0   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?��
=p�   max       @vf=p��
     �  D�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @(         max       @M�           p  M`   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @�L           �  M�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         C�   max         C�      �  N�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�
=p��   max       ?�����m     0  O�                  `   G   �            )   �   !   +   
   	   K      
   ,   %            #   
   D   �      ;   B                           $      
      �            �   
         Q      HO�O�6�O�N"<cO<4�P� �O��.P�a%N��8M�5&M�5�PQ�P�Y�O�5O�
*N�@N���P�+N�~N��PO��O��]N[�O2%PsXN��"P2��P�%O!�PE-#P�6O���N�K�O �KMM��9N��7OV&�O�DO� ?N�lO
�N}PGL�N��GNU�NOO���Nt[N��N16O�qN!@�O`3��������C��D���#�
��o�D����o;o;D��;��
;ě�;ě�;�`B;�`B<e`B<u<�C�<�C�<�t�<�t�<��
<�9X<�j<�j<���<���<���<�<��=o=�P=49X=8Q�=H�9=L��=P�`=Y�=Y�=e`B=u=��=��P=���=��w=���=� �=�^5=\=Ƨ�=��#=��#=��m>V>V��������������������qty��������������zvqpjhknt������������tpz�����������
#09870#
��������)BMLSKOD5�����������������PUa����)58+!�����zcPYU[cgtw����{tgd[YYYYhhnt���}uuttrhhhhhhh��������������������XWX[mz����������zmaX
B[����������gB
/;HTamma[T;/""/"#*<Uaoz�y�znaH</#")0355650,)$����
 "
�����������/<BEC</#
��������������������������������������������������
!%&
����������
#/AF</#
���YYfv������������tg^Yomot����}toooooooooosrst������������{ts)5Ngt����tk`[5- ��������������������36BCXh�������thV;773�����������������������������������������|z��������������<DXmz���������zm^TQ< /BHTaivvpm]H;/"����	
������������������������������ $��

�����������������������������������������������������������
������tmhcOB6))18BO[`ht�{������������������_ZX[_amz~�����zrma__���������������������z������������������869<?HMUX]]URHC<888884:<IROIH<8888888888Z[^ehmtyxtihd[ZZZZZZ��������

�����!#/5785/*#!!!!!!!!
	�

CCEHUXWUUHCCCCCCCCCC��������#"�����������������������������

������čĚĦĳĸĿ����ĿĿĳĦğĚĎčąĆčč�N�[�g�l�j�f�[�M�B�:�5�)�����!�5�B�N�n�zÇÓàìîìàÓÓÇ�{�z�n�m�k�j�k�n�;�H�O�N�H�?�;�8�0�9�;�;�;�;�;�;�;�;�;�;���������������������x�l�x�|������������ŇŔŤŹ��������Ŕ�{�0�������������#�fŇ�ּ������"�!������ּҼǼļɼѼ��������������	�������Z�H�"�.�Z�x���׾����������������������������������������������������������������������������������'�)�'������������������5�A�`�i�h�s�j�Z�5�(�����������B�[�vĔēĀ�L�G�O�B�<�)�� �����(�B�����u�`�[�X�P�C�;�5����H�T�a�m�z��������������������������������������������������������������z�n�z������������"�$�����������������������������������y�`�;�%�����.�T�`�y���/�<�H�Q�H�@�<�3�/�-�-�.�/�/�/�/�/�/�/�/�'�3�?�;�3�3�'������'�'�'�'�'�'�'�'�(�A�M�R�l�m�h�Z�M�4��н����Ľнݽ��(�<�H�a�n�zÉËÇ�y�n�a�R�W�P�H�'���#�<�y���������������������������m�h�a�d�m�y�:�F�S�W�U�S�F�<�:�3�:�:�:�:�:�:�:�:�:�:���������Ľ½��������������z�{�������������(�3�9�9�3�7�A�<�5�(���ݿֿ��ѿ�����(�4�A�D�A�@�4�,�(����	��������žξʾ�������������Z�B�>�@�M������������!�5�F�I�9���ֺɺ����������Ⱥֺ���(�5�A�N�S�X�N�A�5�(����������ìù��������������ìà�z�g�i�zÇÏÕâìčĦĿ��������ĸĦĚĈ�}�w�n�k�l�tĀāč��#�0�=�I�Q�U�S�I�<�0�#� ���
�
��
��`�m�y�����y�m�`�T�N�T�W�`�`�`�`�`�`�`�`ÇÓàìì÷ù����ùìàØÓÇÆÄÆÃÇ�uƁƊƈƁ�u�h�h�h�s�u�u�u�u�u�u�u�u�u�u�������������������������������������������������������������������������������������������������������������������������y���������������������������y�w�u�u�x�y���������~�u�o�r�~�����������ʺʺº�����������!�+�-�2�-�!����������������ŔŠŭŹ����������źŹŭŪŠŘŔőőŔŔ�<�>�B�H�N�H�E�<�/�#��#�-�/�9�;�<�<�<�<��4�A�L�L�:�'����л����������л�����(�4�>�A�F�A�:�4�(�����������ֺ�������ֺֺӺֺֺֺֺֺֺֺֺֺּ4�=�@�M�W�M�J�@�4�0�'�'�'�,�4�4�4�4�4�4D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�DD�D�D�ǈǔǡǬǫǡǔǈ�{�w�{Ǆǈǈǈǈǈǈǈǈ�H�<�0�#�����#�0�<�I�I�I�H�H�H�H�H�HŭŹ��������ŹŭŬšŭŭŭŭŭŭŭŭŭŭ¦²¿��������������¿²£ECEPE[E\EfE\EPECECE9ECECECECECECECECECECEuE�E�E�E�E�E�E�E�E�E�EuEsEnEnEkEiEfEiEu 6 G A D \ V < I 5 l Q 6 l 0 O j . 3 n Q Z I [ 1 = V I K - X > 0 V L ? R @ L $ . = T " w = " < 2   X A Y $ a 4    -  [  O  <  �     �  �  �  J   �  n  �  O  �  �  �  �  U  �    �  �  0  x  �    I  �  n  X  5  �  �  c    1  �  �  a  -  �  3  �  �  �  <  e    x  �  K  �  I  ּu�ě�;ě��o;�o=�j=�7L>bN;ě�<t�;�`B=<j>C��=#�
=L��<�j<ě�=�^5<��
<�/=}�=e`B=��<�/=C�=u=\)=�v�>49X=,1=�Q�=��=�7L=P�`=��
=aG�=�%=q��=���=�C�=\=�j=�1=�^5>Z�=��=ȴ9=��>��^=�"�>V>o>P�`>�u>XbB��B5B
:B
��B$�`B��B� B�B	�!BW}B!k_A�T9Bz�A�8�B�B��B��B��B -NB!��B$L�B�<B
W�BqBPlB�#B1�B�B�B:�B�~B cA�[VB�YB!��B(�B�YB��B	�B-8Bg"B)��A��VB��B��BM�B&�gB� BւB.	B�YBj�B�dBo�B,OB�dB@B
?�B
��B%7*B>{B��B�~B	�B>�B!IsA�w?B@�A��mB��B�LB�CB�qB�CB!UB$?�B�eB	��B�B4�B��B9B�MB/JB>�B8]A��KA��B��B"?�B<�B��BX~B'�B,��BEB)��A��ZB?�B+�B@�B&X^B�GB�*B�B�\B rBFPB?�BH
A���A�e^Aɠ�A���@���A�xAu�A�u'AK�@�.�@��A�TA��VA��YA�e�A�,�A�'2Ai�.A�xx?��XA40�AƑ�Ap"i@�TA ��A��vA6k�AD��@L��A�ΣA��A���A��fAj�A�X6B��A���A�U�A�L�AF@h]@bKsA���A��Y@��:A7�@E�@Х}C��B6�A� A���A��C���C�bA���A�r�A��XA�g@��IA�~2AlA�GYAJ��@��@��A���A�z�A�5bA�~�A��A��Ai5vA�z�?��IA7 9AƉ�Am
p@��A"�A���A6�iAD�@K�A��;A͐nA�waA�yAiA�dBƤA��A�(vA�m�AO�@�+@c�BA�<MA�@���A6�|@C�m@�BC���B>+A�DA��?A�p�C���C��                  a   G   �            )   �   "   ,   
   
   K      
   -   %            $      D   �      ;   C                            $            �            �            R      I                  C   !   Q            '   A   '   !         '         /   %            /      /   +      /   #                                       /                        !                        +      )                                       )                              %                                          -                        !      N���O"�O�yN"<cO<4�PΦOdϰPJ�N��8M�5&M�5�N���O��zO��N��N�@N���Of��N�~NNt�O��N���O<PN[�O2%O���N��"OS��O��=O!�P�oO��.O���N�K�O �KMM��9NW4�OV&�N��'O� ?N�lO
�N}P�N��GNU�NOO&�Nt[N��N16O�1N!@�OQ�y  �  0  �  �  6  p  	�  	z  �  F  :  �  �  }  D  �  �  	T  �  =  �  �  /  �  �  *  g  |    �  J  	�  �    �  �  �  =  7  �    �  �  �  �  �  �  �  �  _  b    P  �  ż�h���㼃o�D���#�
=�P<��
=�hs;o;D��;��
<�`B=��`<u<���<e`B<u=D��<�C�<��
<ě�=#�
<���<�j<�j=+<���=m�h=� �<��='�=aG�=49X=8Q�=H�9=L��=P�`=]/=Y�=q��=u=��=��P=���=ȴ9=���=� �=�^5>�=Ƨ�=��#=��#=��>V>hs���������������������|~�����������������qkhlot����������tqqz�����������
#09870#
��������)-44?@<5)���������	����������������������������YU[cgtw����{tgd[YYYYhhnt���}uuttrhhhhhhh��������������������fdeimyzz������zmffff3,*,15BN[gv}~tg[NB3	"(/;DJMONH>3/"--/0<HSU\UTH</------)0355650,)$����
 "
���������
#/7<=<;7/#
����������������������������������������������
"$$
�������


������f^^gmu����������tgfomot����}toooooooooosrst������������{ts"&5BN[alib[XB51)"��������������������d`_ahnt����������thd������������������������������������������������������������[UU\imz���������zmd[ /BHTaivvpm]H;/"����	
������������������������������ $��

����������������������������������������������������������������������tmhcOB6))18BO[`ht�{������������������_ZX[_amz~�����zrma__����������������������������������������869<?HMUX]]URHC<888884:<IROIH<8888888888Z[^ehmtyxtihd[ZZZZZZ��������

�����!#/5785/*#!!!!!!!!
	�

CCEHUXWUUHCCCCCCCCCC��������#!�������������������������������

����ĚĦİĳĿ����ĿļĳĦģĚĐčĈčĒĚĚ�B�N�[�^�_�^�[�R�N�B�;�5�)�'� �#�)�,�5�B�n�zÇÓàëìììàÓÒÇ�z�n�l�j�k�n�n�;�H�O�N�H�?�;�8�0�9�;�;�;�;�;�;�;�;�;�;���������������������x�l�x�|�������������0�I�n�{ŌřśŏŇ�n�b�
�������������#�0�����
���������޼ּռѼμԼּ�������������������������������y�g�O�J�T�������������������������������������������������������������������������������������'�)�'�����������������(�5�A�I�N�N�N�D�A�5�(���������6�B�O�[�h�n�y�|�z�u�h�[�O�B�8�3�0�/�/�6�T�a�m�y�z�|�w�m�a�T�H�;�3�"����/�H�T�������������������������������������������������������������z�n�z������������"�$���������������������m�y�����������y�m�`�V�T�G�;�8�@�O�T�`�m�/�<�H�Q�H�@�<�3�/�-�-�.�/�/�/�/�/�/�/�/�'�3�<�9�3�.�'������'�'�'�'�'�'�'�'�(�4�M�Z�e�g�a�M�A�4���ݽ����нݽ���(�a�m�n�z�z�z�z�n�a�U�S�H�E�H�U�V�a�a�a�a�m�y�������������������������|�y�q�l�k�m�:�F�S�W�U�S�F�<�:�3�:�:�:�:�:�:�:�:�:�:���������Ľ½��������������z�{����������������(�1�.�.�,�(������ݿϿ׿ݿ�����(�4�A�D�A�@�4�,�(����	������f�s���������������������s�f�[�Y�Z�a�f�ֺ������!�!�������ֺɺĺ��ºɺ���(�5�A�N�S�X�N�A�5�(����������ù����������������ìàÓ�z�r�p�qÇÓàùčĚĦĳĿ������ĹĳĦĚčĆā�x�w�zāč��#�0�=�I�Q�U�S�I�<�0�#� ���
�
��
��`�m�y�����y�m�`�T�N�T�W�`�`�`�`�`�`�`�`ÇÓàìì÷ù����ùìàØÓÇÆÄÆÃÇ�uƁƊƈƁ�u�h�h�h�s�u�u�u�u�u�u�u�u�u�u�������������������������������������������������������������������������������������������������������������������������y������������������������y�x�y�y�y�y�y���������~�u�o�r�~�����������ʺʺº�����������!�+�-�2�-�!����������������ŔŠŭŹ����������źŹŭŪŠŘŔőőŔŔ�<�>�B�H�N�H�E�<�/�#��#�-�/�9�;�<�<�<�<�'�4�?�E�E�A�4�'�����л������ܻ���'��(�4�>�A�F�A�:�4�(�����������ֺ�������ֺֺӺֺֺֺֺֺֺֺֺֺּ4�=�@�M�W�M�J�@�4�0�'�'�'�,�4�4�4�4�4�4D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�ǈǔǡǬǫǡǔǈ�{�w�{Ǆǈǈǈǈǈǈǈǈ�H�<�0�#�����#�0�<�I�I�I�H�H�H�H�H�HŭŹ��������ŹŭŬšŭŭŭŭŭŭŭŭŭŭ¦²¿��������������¿²¤ECEPE[E\EfE\EPECECE9ECECECECECECECECECECEiEuE�E�E�E�E�E�E�E�E�E�E�EuEtEoEoEmEiEi / C > D \ e + . 5 l Q ! * * ( j . ' n P ^ - ^ 1 = ! I # ' X 8 ' V L ? R @ E $ 2 = T " w : " < 2  X A Y $ a 0    �  W  7  <  �  �  �  �  �  J   �  
  K  d  �  �  �  �  U  v  ]  �  �  0  x  b    �  	  n  �  U  �  �  c    1  �  �  �  -  �  3  �  �  �  <  e  [  x  �  K  �  I  �  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  }  �  �  �  �  �  }  f  I  &  �  �  �  v  D    �  �  �  D  �        '  -  /  /  +  %      �  �  �  �  �  �  5  �  �  �  �  �  �    Y  0    �  �  f  .  �  �  �  S  $  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �      !  6  %        	  �  �  �  �  �  �  �  �  �  �  �  �  �    �  �  -  j  �    ;  `  p  g  E    �  z    �  �  �  �  �  �  �  	.  	X  	v  	�  	�  	�  	�  	m  	?  �  �  M  �  #  j  �  x  �  n  5  �  �  �  	0  	?  	>  	M  	p  	s  	:  �  r  �  I  �  d    `  �  �  �  �  �  �  �  �  �  u  c  M  6     	       �   �   �  F  c  �  �  �  �  �  �  �  �  �  l  Q  3    �  �  �  �  �  :  3  -  '  !            �  �  �  �  _  <    �  �  �  K  �  �  �  �  �  J  �  �  �  �  �  �  �  �  J  �  �  �  �  z    �    �  �  �  �  Q  �  �  �  �      �  �  
�    r  �    J  m  z  z  j  M  '  �  �  �  �  n  ?  �  n  �  T  
  �    p  �  �    ,  A  D  >  -    �  �  s    �  \  �  {  �  p  _  N  >  2  )  !        �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  ~  o  ^  H  1    �  �  �  �  �  �  :  �  "  �  �  	'  	G  	T  	5  �  �  D  �  U  �  �  �  �  6  �  �  �  �  �  �  �  �  �  �  �  �  �  �  |  t  l  d  \  T  6  9  ;  <  =  <  <  ;  ;  8  4  .  '         �  �  �  �  �  �  �  �  �  �  p  Y  ^  A    �  c    �  ~    n  �   �  *  n  �  �    N  t  �  �  �  �  �  �  q  .  �  b  �  L  �  �       )  .  ,       �  �  �  �  t  �  �  	    0  ?  N  �  �  �  �  �  �  �  �  �  �  �  �    :  W  r  �  �  �  �  �  �  �  �  �  �  t  f  S  ?  )    �  �  �  k  I  6     �  u  �  �    *  (      �  �  �  �  `    �  w    �  &  �  g  Z  M  =  -      �  �  �  �  �  y  T  @  3    �  �  �  �    W  �  �    8  _  t  |  x  k  R  !  �  j  �  b      �  6    �  9  �  �       �  �  a  �  q  �  �  f  	�  F    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  u  L     �     2  F  J  D  ,  �  �  c    �  �  �  �  �  �  j  �  /  �  Z  �  	C  	u  	�  	�  	�  	�  	w  	Q  	  �  g    �  �  >  P  4  e  �  �  �  �  �  �  v  \  S  R  U  P  ;    �  �  I  $  �        	    �  �  �  �  �  �  �  �  �  �  ~  g  P  8     	  �  �  �  y  J    �  �  L  �  �    �  5    �  �  X    �  �  �  �  �  �  �  �  �  �  �    p  ^  K  8  %    �  �  �  �  �  �  �  �  �  �  �  �  l  R  4    �  �  �  �  o  J  %  7  9  ;  =  5  +  "      �  �  �  �  �  �  ~  e  K  0    7      �  �  �  �  �  �  �  �  �  �  `  /  �  �  V  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  r  g  Q  8    2  h    �  �  �  �  s  W  7    �  �  {  7  �  �  '  �  m  �  �  �  �  h  M  /    �  �  �  �  l  >    �  y  $  �  P  �  I  �  {  e  O  7    �  �  �  �  |  V  2    �  �  �  y  :  �  �  �  �  �    M  H  @  8  /  $      �  �  �  �  f  �  ^    T    �  o  ?  �  �  <  �    a  �  �  
�  	�  �  �  T  �  �  �  �  s  g  Z  M  ?  1  "    �  �  �  z  '  �  i  �  t  �  �  �  m  V  @  )    �  �  �  �  �  ~  m  Z  @    �  Q  �  �  �  �  �  �  �    k  U  )  �  �  �  q  M  (    �  �  �  �  6  l  �  �  �  �  n  #  �  �    �  �        
�  �  _  G  .    �  �  �  �  �  d  B      �  �  �  �  �  E  �  b  0    �  �  �  u  Q  )  �  �  �  w  ?  �  �  g    �  M       �  �  �  �  �  �  �    v  t  q  r  z  �  �  �  �  �  P  >  -      �  �  �  O  �  �  %  
�  
  	p  �  �  �  g    �  �  j  P  6    �  �  �  �  �  q  X  A  �  �  >  �  g  �  �  �  �  l  6  �  �  u    x  �  ,  j  �  
�  	�  �  v  �  �