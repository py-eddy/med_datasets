CDF       
      obs    9   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?Ǯz�G�      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M̂7   max       Pƒ�      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       �L��   max       =�{      �  t   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?��R   max       @E���R     �   X   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ���Q�    max       @vo
=p��     �  )@   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @!         max       @N@           t  2(   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @ȵ        max       @���          �  2�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �D��   max       >R�      �  3�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A��n   max       B/
�      �  4d   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�2   max       B.�m      �  5H   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >�,�   max       C�+�      �  6,   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >��   max       C�0#      �  7   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �      �  7�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          K      �  8�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          9      �  9�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M̂7   max       Pt�      �  :�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�*�0�   max       ?��Y��|�      �  ;�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       �L��   max       =�      �  <h   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?@        max       @E���R     �  =L   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ���Q�    max       @vl�\)     �  F4   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @$         max       @N@           t  O   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @ȵ        max       @��           �  O�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         ?�   max         ?�      �  Pt   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�hr� Ĝ   max       ?���Q�     �  QX                  (         
      	      $   �      0         3         B         %               o      *      �                           5            [         A      E   E      E      N'��N���N�`O#��O��{POE�*N�M���O�4N�a`N�)7P*Pƒ�O	��O�b7N��GN�^oO�N���N�UPv��Oc
@N�;O�#�N�u{O��TN��nOq-�Pw[1OduoP��Os�P%P�OnN�S�NG�O+��N��fO<��O"-O.�O��CN�N�Nb'qN�opO�K�N���N�w1O�=�O��4P(gO��M̂7O��wN��nO,��L�ͽD���'�9X��o�t��t��o�ě��ě��D��:�o:�o;D��;��
;�`B<o<t�<t�<#�
<49X<D��<�C�<�t�<��
<ě�<ě�<���<���<�/<�h<�=+=C�=C�=t�=�P=��=#�
=,1=8Q�=8Q�=@�=D��=H�9=T��=Y�=q��=q��=u=y�#=�%=�+=�C�=��=��=�{��������������������sjist�������ttssssss�����������������������������	���������A=JMNUbnz�����}nbUIA��� )5MNGKHB5)
��&()6BOirkheih[OEB6)&`^]`abdnx{�|{zvnb``��������������������gnnu{������������{ng��������������������������������������������)5<Napl[N5)��"EF[������������t69537<HOUanvnldaUH<99����������
����f^]gt}������xtngffff�������������������GCCFO[ht������{thb[G����������������������������������������//-/5B[��������[NB5/jdbbmx����������zwmj�����������������������
#/HUij^</#���wtoqz������������zww	*56BNRTSOB51�������������������������������uz����������������u2+,.257@BEK[hig[NB52���� 6OWh{bTK6)��4/.15=BN[]dfmng[NB54VRT[bt�����������g]Vc\^bggpt��������tgcc�������



	������	 %trqtu�������������tt�����	�������������������������������� ���������

������� �#/DU_`WZUL/# b`fhnt��������thbbbb���������������������xtphhghipt������������������ 

�����),,)&$1145@BNX[\[TNB<51111-##$+;HTaflopmdIH;1-������	����������kebn��������������rk�������	��������������������������������
)//+#
�����%),6@BKOPOKFB<61*)%%�������������������������������������������������������������a�n�zÇÇÌËÇ�z�q�n�f�a�a�a�a�a�a�a�a�{ŇŔŘŝŔŇ�{�n�i�n�t�{�{�{�{�{�{�{�{���������������������������������߻����ûл�������ܻû������x�q�s�x��������"�;�Q�T�Z�[�;�.�	���ؾʾ������������������������������y�v�{�y�������������Ľнݽ���� �����ݽнʽĽ���������²³¿����������¿²²°²²²²²²²²���������üǼ�����������������������������*�6�>�C�G�G�C�6�*�����������	����"�%�'�$�"���	������	�	�	�	�\�hƁƊƙƭƱƧƛƁ�h�\�P�O�D�A�B�;�C�\�ü�f���ʼʼټټ�����Y����û��ƻ�����������������������������������(�A�L�Z�j�s����x�Z�7�(��
� ������;�H�T�_�]�T�Q�R�H�;�2�4�/�,�/�8�;�;�;�;���������������������������������������žM�Z�f�s��������������M�4�+�(���(�.�M����������������������������������������6�7�B�B�O�R�O�E�B�6�)�(��!�)�2�6�6�6�6�Z�������������������s�g�Z�R�I�G�O�T�U�Z����������������������������ƿƾ�����̻�������������������������������������������(�3�;�<�9� ����������������������������������������s�n�s���������M�Z�f�l�s���s�R�M�A�4�(����*�4�A�M�5�A�N�P�N�B�A�9�5�,�(�$�����(�0�5�5������(�5�A�G�F�5�(��������������ʼټ��׼����f�M�4�!��&�'�,�@�r�����������$�6�=�I�R�I�=�0�$������������"�;�G�X�P�;���"��"�	��׾ȾþǾ�	�"¦µ¿����¶²¦�t�j�n�q�u������6�B�P�U�W�V�O�6��������������������)�5�B�B�N�W�R�N�B�5�)��������#�&�/�<�?�H�T�J�H�<�/�#������#�#�#�������������������������������������������������������������������������������������������������������������������������`�m�y�������������y�o�n�m�b�T�Q�O�K�T�`�;�=�;�:�;�>�>�;�/�"������	���"�3�;�����������������~�r�e�Y�Q�L�Y�r�~�������4�A�Z�f�r�y�y�s�f�Z�A�4�(�������4�!�-�:�<�F�P�S�F�A�:�3�-�!����!�!�!�!�
�	����������
�������
�
�
�
�
�
�y�����������������ĽǽƽĽ������������yD�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D���#�0�<�=�=�A�<�0�(�#����������I�U�b�f�n�o�s�n�i�b�_�U�Q�K�I�F�I�I�I�I�������
��&�,�)� ��
��������ĳĪıĿ�̽������������������y�l�f�`�Z�S�S�`�l�y����������������ܹϹù������������ܹ��U�_�`�^�Y�Q�D�/�#��
�����������#�<�B�U�	�����	��������	�	�	�	�	�	�	�	�	�	E�E�E�E�E�E�E�E�E�E�E�E{EwE{E�E�E�E�E�E�����������������غ��������ÇÓÜàäççàÓÇ�z�n�l�j�l�n�s�v�zÇ 8 M E  ; A G W o 1 ' L ; h J / Q % & 9 % 3  X S y 2 s 7 U < \ 6  D 7 E @ / 7 ] m  > t D 9 B N 0 % B S X A 7 ;    J  �  �  X  �  �  �  	  I  1  �  �  �  �  I  �  �  �    �  �    �  5    6  Z  �  �  j  �  @  �  �  R      �  �  �  �  �  �  �  �    )  �  �  �  ?  �  �   �  �  �  z�D���\)��㻃o<u<��;ě�;��
;�o;�`B;ě�<o=t�>@�<���=aG�<T��<��
=q��<�9X<���=��-=<j<�1=e`B<�h='�<��=t�>I�=L��=�hs=m�h>R�=L��=@�='�=T��=@�=}�=m�h=e`B=���=aG�=u=�C�>t�=�o=��=��=�->+>	7L=�t�>bN=�{=�hB��B
 �BBe�B'�B�2B�EB(/
B�FB)ZB03B6B@B2�B36B#5�B	�B��B�B�B�DB	��B >�B"��B�B ?�B=B��BzB� B��B�B<OB
|RB	�@B'B/
�B
�jB�%B��B�[B#��BDB[B�EB��B��B3�B�,A��nB,�bB��B��B%BǚB�B!��B�B	��B2�BG�B'��B?�B��B(B�B�B)�9B8�B?qB�B?vB?FB#@ B	��B��B<�B��B��B
�~B B�B"��B��A��B��B��B*BG�B8.B�.B�LB
U�B	�B9B.�mB
� B�_B�^B��B$X(BĬB{�B��BF�BÝB?�B:�A�2B,��BλB��B7�B�BҨB!ӷA��A�/"A�#:A���@��A\�VAr�KA)�nA���@��A�v�A�<�BmP@�lLA��8A8�(A�i�A�~�AB�A�ÍA׏�A�O{B$�@���A�,A���A<�mA��aA� @��gB	N�A\�NA�N�A�^7A�vA�6�At�VAs�A�ǰAk3�A��L@��A<X�@wWA��A!��C��>A�+A��
A�/A�O>�,�A�/OA�xtC�+�@T-A�~�A�~WA�R�A��A�y�@��A^�$As5A)$�A�x`@���A�dA���BI}@�Q�AҀhA8��A���A�)�ACyAЌ�A�x�A�|�B9@�܍A�s8A���A>��A�W>A���@��B	=A_��A�m�AՁ�A�u�A�{�At�As�A�dAk;�A��S?�t�A= /@�EXA��BA#OC��A�yHA�A�`A��>��A�{�A�3sC�0#@X�JAɁ�                   )               
      $   �      1         3         B         %               o      *      �                           6            \         A      F   E      E                      %   )                     '   K      %         !         /         #               9      9      %                                                !   !   '   %                              !                              !                  '                        9      +                                                         !   '   !            N'��N}��N�`N�sO�pTO�HTO,�N��eM���N��NjxN`F�O��HOq��N"�WO��GN��GN�^oO*/�NU�.N�UP9�=O��N�;O��2N�u{O��TN��nOq-�Pt�OT;O�{�OKIO�}�N�ݧNտ�NG�O+��N��fO5�OиO.�O���N�N�Nb'qN�opN��N���N�w1O1/|O��4P(gO��NM̂7Od��N��nO,��  �  �  �  �  l    e  �  �    4  (  x  E  �  ~  D  �  �  �  �  $  m  �  �  �  �  �  �  �  �  �  �  I  �  �  �  N    �  ~  i  s  �  �    >  �  �  	�  �  	�    �  
�    ��L�ͽ<j�'�o�o%   �o�ě��ě���o%   ;o<T��=�<e`B<T��<o<t�<��<T��<49X<ě�<���<�t�<ě�<ě�<ě�<���<���<�`B<�=C�=t�=�`B=��=�P=�P=��=#�
=8Q�=<j=8Q�=�%=D��=H�9=T��=�^5=q��=q��=�9X=y�#=�%=�t�=�C�=�Q�=��=�{��������������������kktz�������tkkkkkkkk����������������������������������������LJLQUVZbnw}����{nbUL��)5FGE>A;5)
'))6BO[gpjhb[OGB6*)'b_^abbgnu{~}{zxtnbbb��������������������rr{{������������{rr����������������������������������������)5BNU]`[WN5)<;<?CIO[hqtvutng[OB<99<HPUWUNHD<99999999��������
�����f^]gt}������xtngffff�������������������YXX[dht�������yth^[Y����������������������������������������@:67BNg����������[J@mlmqz����������zummm������������������������
#/<HW`UH</#�wtoqz������������zww	*56BNRTSOB51�������������������������������u{����������������u3--/359ABI[ghf[NFB53�����%6BUWNE)��10256@BHN[bdkgd[NB71efint�����������tphe`aegqtx�����ytmg````�������	

������	 %trqtu�������������tt�����	��������������������������������������

�������	#/<GOOLH></#	b`fhnt��������thbbbb���������������������xtphhghipt�����������������

������),,)&$1145@BNX[\[TNB<511116559;HTT_adeda]TH;;6������	����������kebn��������������rk���������	������������������������������
"&'&#
	����%),6@BKOPOKFB<61*)%%�������������������������������������������������������������n�zÅÇÉÇÇ�z�t�n�g�c�n�n�n�n�n�n�n�n�{ŇŔŘŝŔŇ�{�n�i�n�t�{�{�{�{�{�{�{�{�����������������������������������߻������ûлܻ����ܻл����������{�|�������	�"�.�;�H�N�Q�Q�;�.�"�	����ݾҾӾ���������������������������~�|���������������Ľнݽ����������ݽսнĽ���������²³¿����������¿²²°²²²²²²²²���������ü�������������������������������*�6�7�=�8�6�*�&������������	��"�$�#�"��	���	�	�	�	�	�	�	�	�	�	�h�uƃƎƕƔƔƉƁ�u�h�_�\�Y�L�K�P�R�\�h������'�4�<�F�H�@�:�4�'�������������	���������������������������������(�A�Z�c�n�q�f�Z�M�A�4�(��������(�;�H�T�_�]�T�Q�R�H�;�2�4�/�,�/�8�;�;�;�;���������������������������������������žZ�f�s�������������s�f�Z�S�M�I�D�M�R�Z�����������������������������������������6�7�B�B�O�R�O�E�B�6�)�(��!�)�2�6�6�6�6�g�s�������������������s�g�Z�S�O�Q�P�Z�g��������������������������������������ٻ���������������������������������������������(�.�5�7�3�'����������������������������������������s�n�s���������M�Z�f�l�s���s�R�M�A�4�(����*�4�A�M�5�A�N�P�N�B�A�9�5�,�(�$�����(�0�5�5������(�5�A�G�F�5�(��������������ʼټ��ּ����f�M�4�"��'�'�-�@�r�����������$�0�=�I�I�=�/�$��� ����������"�.�;�G�U�T�M�.���	����׾Ǿ;��	�"¦°²¿��¿½³¦�v�t�q�s�w���)�6�<�C�E�A�6�)�����������������)�5�:�B�F�B�=�5�)������������#�/�<�>�H�O�H�<�:�/�#���������������������������������������������������������������������������������������������������������������������������������`�m�y�����������}�y�m�`�T�S�R�O�T�`�`�`�/�;�=�=�;�:�/�-�"���	���	���"�#�/�����������������~�r�e�Y�Q�L�Y�r�~�������A�M�Z�f�h�q�r�n�f�Z�M�A�9�(�&���%�4�A�!�-�:�<�F�P�S�F�A�:�3�-�!����!�!�!�!�
�	����������
�������
�
�
�
�
�
�y�����������������ĽǽƽĽ������������yD�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D���#�0�<�=�=�A�<�0�(�#����������I�U�b�f�n�o�s�n�i�b�_�U�Q�K�I�F�I�I�I�I�������
�����
��������������������彅�����������������y�l�f�`�Z�S�S�`�l�y����������������ܹϹù������������ܹ��/�<�H�U�[�]�[�W�O�/�#��	���������
�#�/�	�����	��������	�	�	�	�	�	�	�	�	�	E�E�E�E�E�E�E�E�E�E�E�E�E�E~E�E�E�E�E�E�����������������غ��������ÇÓÜàäççàÓÇ�z�n�l�j�l�n�s�v�zÇ 8 D E  C > G Z o > $ = ; @ . ) Q % " B % 2  X K y 2 s 7 U 6 ` 0  % / E @ / ( 5 m  > t D ( B N & % B O X < 7 ;    J  �  �  �     �  �  �  I  �  t  v  a  �  9  s  �  �  g  j  �  +  5  5  z  6  Z  �  �  `  �  �  �  �  �  �    �  �  5  *  �  �  �  �    �  �  �  t  ?  �     �  �  �  z  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  �  �  �  �  �  �  �  �  �  �  �  �  �  �  ~  u  m  d  [  R  �  �  �  �  �  �  �    t  h  Z  L  >  /       �  �  �  0  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  v  i  \  O  B  �  �  �  �  �  �  �  �  �  �  �  �  �  �  g  0  �  }  "  �  ;  X  j  l  k  l  k  i  g  a  X  I  '  �  �    :  �  1   `  �  �  �          �  �  �  �  v  c  @    �  �    �  #  3  [  _  T  E  4  &      �  �  �  �  �  e  H  "  �  S    �  �  �  �  �  �  ~  g  G  !  �  �  �  A     �  x  @  8  f  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �        	     �  �  �  �  �  �  �  �  �  �  �  �  w  d  �  	       )  .  2  0  )        �  �  �  �  �  �  �  �      $  "    �  �  �  �  �  �  �  e  J  -     �   �   �   �  �  �    8  O  d  t  w  o  \  A  !    �  �  �  e    �  �  �  
6  :  �  �    <  v  �  �  '  D  @    �  �  �  
V  �  �  I  T  e  z  �  �  �  �  �  �  �  �  }  �  r  �  �    z   �  �  `  |  |  l  L  (    �  �  q  $  �  ^  �  {  3  �  �   f  D  <  4  -  '  (  (  )  )  &  #  !  !  "  $  %  ,  4  <  E  �  �  �  �  �  �  �  �    e  H  '      +      �    *    �  "  O  u  �  �  �  �  �  �  �    N    �  G  �  �  �  |    �  �  �  �  �    w  n  c  X  P  H  C  =  8  5  4  4  �  �  |  r  b  I  #  �  �  ~  ?    �  {  =    �  N  �  q  �  �  
  !  "    �  �  �  t  M  &  �  �  R  �  V  �  �  k  �    *  I  _  k  j  \  D  &  �  �  �  ^    �  ]  �  �  U  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  u  G    �  -  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  i  C     �   �  �  �  �  �  �  �  �  �  �  �  �  �  �  v  3  �  �  (   �   \  �  �  �  �  �  w  h  Y  N  H  B  <  ,      �  �  �  �  �  �  �  �  y  p  f  \  T  K  B  6  (      �  �  �  e     �  �  �  �  �  �  �  o  ;    
�  
�  
l  	�  	G  �  �  >    U  D  �  �  �  �  �  y  _  C  '    �  �  �  u  >    �  g  �  ;  _  �  �  �  �  �  t  [  T  �  �  �  �  v  =  �  p  �  D  f  w  �  �  �  �  �  v  ^  A    �  �  ~  :  �  �  [    �  �  �  ?  �  �  )  �  �    A  C    �    r  �  �    8  	  	  �  �  �  �  �  �  �  �  �  �  �  �  }  Z  3  
  �  �  �  �  �  �  �  �  �  v  b  M  8  #      �  �  �  �  �  7  	  �  �  �  �  �  �  �  �  �  �  ~  x  p  g  ^  V  K  A  6  +     N  I  @  6  -         �  �  �  �  �  �  [    �  t  %   �            �  �  �  �  �  �  l  I  '    �  �  �  s  M  3  �  �  �  �  �  �  ~  l  Y  C  %     �  �  �  f  A    �  5  g  |  x  m  _  P  ?  ,      �  �  �  �  f  .  �  �    i  U  A  ,      �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �    I  d  r  s  i  S  2    �  �  _  �  }  �  2  j  I  �  �  �  �  �  }  f  L  3    �  �  �  �  �  �  �  �  �  y  �  �  �  �  s  ^  ?    �  �  �  U  "  �  �    F    �  �    �  �  �  �  r  J  !  �  �  �  }  Z  :    �  �  \  �  9  �  K  �  �  �    ,  ;  >  &  �  �  �  W  �  �  
�  	�  �  g  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  z  q  i  `  X  �  �  �  �  �  �  x  k  _  T  G  9  +      �  �  �  �  �  �  J  �  �  	5  	`  	y  	�  	�  	�  	�  	l  	5  �  E  �    P  �  �  �  �  �  �  �  �  �  �  �  �  �  p  P  (  �  �  _  �     U  	�  	�  	�  	�  	�  	S  	  �  �  �  d    �    �  �  )  [  6  �  
�  
�    
�  
�  
�  
�  
e  
   	�  	u  	  �  C  �    O  Z  �  �  �  �  �  z  q  h  ^  T  J  @  4  &      �  �  �  �  �  �  
�  
s  
�  
�  
�  
�  
�  
�  
`  
  	�  	@  �  r    �      �  '    �  �  �  �  �  �  �  n  M  *    �  �  �  \    �  �  \  �    <  �  �  �  �  l  U  8    �  �  �  K    �  k    M