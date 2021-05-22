CDF       
      obs    5   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?�n��O�<      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M�b�   max       P���      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       �o   max       >C�      �  T   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�p��
>   max       @F*=p��
     H   (   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��33333    max       @vZ�\(��     H  (p   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @-         max       @Q@           l  0�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�e        max       @�r�          �  1$   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       <D��   max       >�(�      �  1�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�G�   max       B0��      �  2�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A��   max       B0��      �  3�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       =g��   max       C���      �  4t   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       =��y   max       C��%      �  5H   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max                �  6   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          =      �  6�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          +      �  7�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M�b�   max       P*vu      �  8�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?���m\��   max       ?��!-w2      �  9l   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       <o   max       >C�      �  :@   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>��Q�   max       @F*=p��
     H  ;   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       �������    max       @vZ�\(��     H  C\   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @"         max       @Q@           l  K�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�e        max       @�#@          �  L   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         ?�   max         ?�      �  L�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�w�kP��   max       ?����l�     �  M�   D   8         =                        #                  D   +   	         U   
      !                           
         E            
   �   2         ,   	   	         O��P���Nz�O�zP1��O51jOC��N puOU��N+��OW��P~�LP��O?�=N��N���OG��O1�Pm1KP.�N	�O��mN�-�Oy<O�*N{��O�Y�N]�	O�iO�yN\=>N�6�N-�<N�P�M�b�Oe�`N�y9O>�O�N�O�cO�4�N�N�!�P&2Oc��OMY:N�+)O�-�N��NNOԊN;M�N�B�N�J�o;o;�`B<o<t�<49X<D��<T��<e`B<�o<�C�<�t�<�t�<�t�<���<�1<�9X<���<���<���<�`B<�`B<��=+=C�=C�=t�=�P=��=��=�w=�w=�w=0 �=0 �=8Q�=8Q�=D��=H�9=Y�=Y�=e`B=m�h=}�=}�=��=�7L=�hs=�hs=��=��=�->C�����������������PYz��������������m^P��������������������67EN[gt������tgd`[N6
5B[t����t[L5������xthf`[[\hjmmt���������������������
+$%/3<HUUaeifbUSH</+X[[hkqtutha[XXXXXXXX��������������������'.B[���������t[N:2)'�����
!/>BHTPH/
���dnnv~������������zod������������������������������������������������������������	"/;??;;/"	��)5B[eh]]YNB)��' ")/=[ht���~wh[B'��������������������������������������������������������������
�����)5:A>>5)qrjt�������tqqqqqqqq�����������
��������������������������������


�����
#%/341/#!
Y[^[VU[goplgd[YYYYYYfgjjtw����|���xtigffrtt��������trrrrrrrr#,0<@B<;20#526COQOHC65555555555!)5BNQZ[YTNB5-)Y[]ggmst������tng[YY����������������������������������������������)10(���������)A64-)
�����������������������������������������������BO[cknmbQB��MLLNR[ht����ztmh[ROM;@ITYamphhfa\YTH;31;��������������������{z�����������������{����������������������'��������*-.5=BDFBA:5********mnnvz���������zuonmm�������������������˾�(�6�A�Z�q�{�p�U�M�A�4�(����������������������������������5�����r�������������ÿ¿����������������������������`�y�������������y�m�`�T�G�?�C�G�T�W�Q�`²¿����������·¦�h�Z�N�[�h�ݿѿĿ��������|�t�y���������ÿĿѿݿ�ݻ��������ûϻջлȻû��������������������!�-�:�>�:�-�)�!�����!�!�!�!�!�!�!�!������� ��������������������������$����������������������������������������������������)�B�Z�h�j�g�[�B���������������������)����(�A�L�g�`�N�5�(���������������(�5�A�N�W�U�N�A�;�5�$���������������������ܹϹ͹Ϲйܹ���àìù��������ùùìààÚÚàààààà�'�4�@�D�?�@�B�@�;�'������������'�a�m�q�x�z�������z�m�a�a�]�V�T�Q�T�X�a�aƎƳ�����	�������ƧƎ�u�d�[�X�a�h�uƎ������������ʾ׾������׾ʾ����\�T�l������������������������������������������z�������������������������z�z�|�{�w�q�z�	��"�.�;�G�M�P�G�@�;�.�"�"���	��	�	DoD{D�D�D�D�D�D�D�D�D�D�D{DqDoDeD_DbDfDo����������� ���������������������������޽�����
�������������������#�/�<�H�U�Y�a�l�q�m�a�<�/������"�#���ʼʼʼɼ¼����������������������������"�.�;�B�G�G�;�;�.�&�"��	�� ��	���"�T�`�m�y������������y�m�`�\�H�G�=�G�M�T�����������������������������������������a�g�m�z���z�q�m�b�a�T�M�H�F�H�J�T�X�a�a�:�F�G�S�S�S�G�F�:�3�.�5�:�:�:�:�:�:�:�:�ּ����������������ּӼ̼Ҽּּּֿm�y�����|�y�v�m�i�l�m�m�m�m�m�m�m�m�m�m�ĿϿݿ��� �-�5�5��������ݿѿÿ����N�T�Z�g�h�s���������s�g�Z�W�Q�N�J�E�N�N�ĽнܽӽнĽ������������z�z�����������Ĺܹ�����޹͹��������|�x�t���������ù����������	��/�2�"��������������������e�r�~�����������������~�v�e�Y�@�8�<�L�e�M�Y�f�r�����r�f�Y�M�H�M�M�M�M�M�M�M�M�M�Z�f�s�����������������s�f�b�Z�T�L�M�����'�C�N�f�q�d�M�@�4�'�����ݻ������������ʼӼмɼ�������������z�s������o�{ŇŒŔśşŔŇ�{�n�`�U�I�D�@�C�I�U�o��(�4�5�9�5�4�(���������������������)�4�5�2�)���������������������*�,�6�C�O�Q�O�C�6�*���	������<�>�B�<�;�4�0�-�#�"�#�#�0�9�<�<�<�<�<�<�0�<�I�R�U�W�U�I�=�<�0�-�0�0�0�0�0�0�0�0����� � ����������޻�������E*E7ECEJEPE\EaEfEaE\EPECEAE7E5E*E*E*E*E* = L A @ = ] ) i . J 6  / t > ! H M ? 8 X A ; * 1 <   : E   Z X 0 L P x f b Z @ R � � ? & i  1 � Y Z . A    �  /  �  M  %  �  �  0  �  O  �  �  u  �    �  �  ?    �  I  #    �  B  �  w  u  0  R  �  �  O  �  -  �    m    &  �  �  \    �  �  �  �  �  �  �  �  	=�+=ix�<D��=C�=�O�<��=C�<�o=,1<���=�w>�(�=Y�<��<��=t�=@�=t�=�j=�C�=t�=<j=49X==49X=�w=�C�=0 �=L��=q��=0 �=,1=8Q�=e`B=<j=aG�=L��=�hs==�t�=���=y�#=�7L>E��=�S�=���=���=�=��
=��=��
=�;d>"��B"�BB ֞B)B	@KB�
B�B"|�BgB�eB��B�iB	A�B��B�%B��B!�;B!�A�G�BAB%�B�DB1B�fB�OB�UB
0�B�B"ؗBotBjB�B	߶B�OB%aB0��B%]B	��B�|BơB�>B,�B��BWB�AB��A�@�B�EB��B�YBt�B@�B��B�/B"��B<B5�B	A�BPpB-B"��B� B��Bp�B�B	@hB�BA B�cB!��B!@�A��BܬB6FB*�B��B=5B�?B��B
��B��B"�By,B@�B	�B	��B��B%@�B0��B�(B
3�B�+B�~B��BE�BըBVuB>xB�7A��B�B:(B�wBB�BN"B�6B�FA8��A�GLAu�LAl&iA��qAs~~@���@p�RA��A2�6A�JA�\�A���A��?',�Ą�@��bA�3SB*AK��A���A��A`��C���A��A/oA�ǒ@�@A_<�Aj]A�=�A��\@0�A �Al��A�&A�BTA"�v=g��A�q@ z,@ݟAD�@�a�@��A��oA6(AӘA��A��A��@��&C���A8��A���Au��Al�.A���As�@���@l��A҈�A3�AЋkA�%|A���A�L?B��A�}�@���A��	B_�AJ��A�gsA���AavC�ƺA�RA.�Aĝ�@���A_�'Aj3A�M�A���@��&A�Am)A���A��MA%=��yA���?��P@ߨ�ABx�@�U�@�WA� A6�vA�^�A��!A�!A�[�@��^C��%   E   8         =                        $                  D   ,   	         U         !                                    E            
   �   2      	   -   
   
            %   =         )                     1   )                  3   +                                                #         #   )   #         +                                 +         %                     !                     %                                                   #            )   !         !                           Oc�^P*vuN,�)N��iO�oN���OC��N puN���N+��N��O���O�JO?�=N��N���OG��N�?uP%�O���N	�O/x�N�-�O��O�*N{��O��jN]�	O�iO��N\=>N�6�N-�<NkZM�b�Oe�`N�y9O>�O�ʟO�cO��N�N�!�O��*O,�2OMY:N�+)O��N��NN([�N;M�N�B�N�J  P  �  :      �  u  /  �  �    �  o  �  �  6  
  �  n  �  #  ]  �  	  5  [    �  �  S  G  �  �  �  s  a  f    �  %    1  �    �  �  E  �  �  a  �  �  	T<���<��
<o<�j<�9X<�C�<D��<T��<���<�o<���>
=q<���<�t�<���<�1<�9X<�/=<j='�<�`B=o<��=e`B=C�=C�=�P=�P=��=�w=�w=�w=�w=8Q�=0 �=8Q�=8Q�=D��=�o=Y�=]/=e`B=m�h=ě�=�\)=��=�7L=�t�=�hs=��P=��=�->C���������������������hckz�������������th��������������������US[^grtzztg[UUUUUUUU)5BN[kyzt[NB5g__`ght�����}tmhgggg��������������������
.,/2<HHTUWUPH<4/....X[[hkqtutha[XXXXXXXX��������������������?<<@N[gt������tg[NF?����
#/59<AGHB</#	�dnnv~������������zod������������������������������������������������������������	"/;=;3/"			)5BNWXTPKB5)	:898;CO[hmruvrnh[OD:���������������������������������������������������������������


 ����)5:A>>5)qrjt�������tqqqqqqqq���������	���������������������������������


�����
#$/231/)#
Y[^[VU[goplgd[YYYYYYfgjjtw����|���xtigffrtt��������trrrrrrrr#0;800#526COQOHC65555555555!)5BNQZ[YTNB5-)Y[]ggmst������tng[YY����������������������������������������������)10(���������)321,)$	�������������������������������������������  6BO^ccXOB6 NOSY[ht{����~vth[SON;@ITYamphhfa\YTH;31;��������������������|���������������������������������������	��������*-.5=BDFBA:5********mnnvz���������zuonmm�������������������˾�(�4�A�M�Z�_�e�Z�M�A�4�(�����������������������������s�Z�5�&�&�.�F�Z���������������������������������������������m�y�����������y�m�a�d�i�m�m�m�m�m�m�m�m¦²¿����������½²¦�q�f�f�k�r¦�������������Ŀ����������������������������������ûϻջлȻû��������������������!�-�:�>�:�-�)�!�����!�!�!�!�!�!�!�!������������������������������������$���������������������������������������������������������)�6�H�O�R�Q�K�B�6�)�����������������(�5�A�R�X�T�N�A�5�(��������������(�5�A�N�W�U�N�A�;�5�$���������������������ܹϹ͹Ϲйܹ���àìù��������ùùìààÚÚàààààà�'�4�@�D�?�@�B�@�;�'������������'�a�k�m�u�y�y�x�m�h�a�_�X�T�S�T�[�a�a�a�aƎƚƧƳ������������������Ǝ�u�k�f�f�uƎ�����������ʾվܾݾ׾ʾ��������z�r�w��������������������������������������������������������������������������~���������	��"�.�;�G�M�P�G�@�;�.�"�"���	��	�	DoD{D�D�D�D�D�D�D�D�D�D�D�D|D{DoDmDkDnDo����������� ���������������������������޽�����
�������������������<�H�U�X�a�k�p�l�a�H�<�/������#�/�<���ʼʼʼɼ¼����������������������������"�.�;�B�G�G�;�;�.�&�"��	�� ��	���"�T�`�m�y��������~�y�m�`�]�T�I�G�?�G�O�T�����������������������������������������a�g�m�z���z�q�m�b�a�T�M�H�F�H�J�T�X�a�a�:�F�G�S�S�S�G�F�:�3�.�5�:�:�:�:�:�:�:�:�ּ׼��������ּռμּּּּּּּּֿm�y�����|�y�v�m�i�l�m�m�m�m�m�m�m�m�m�m�ĿϿݿ��� �-�5�5��������ݿѿÿ����N�T�Z�g�h�s���������s�g�Z�W�Q�N�J�E�N�N�ĽнܽӽнĽ������������z�z�����������ĹùϹܹ����ֹܹĹ����������������������������	��/�2�"��������������������e�r�~�����������������~�x�r�Y�@�9�=�L�e�M�Y�f�r�����r�f�Y�M�H�M�M�M�M�M�M�M�M�M�Z�f�s�����������������s�f�b�Z�T�L�M����'�7�D�S�V�R�@�4�'�����������������ʼ̼˼ļ�������������������������o�{ŇŒŔśşŔŇ�{�n�`�U�I�D�@�C�I�U�o��(�4�5�9�5�4�(���������������������)�3�4�1�)���������������������*�,�6�C�O�Q�O�C�6�*���	������<�<�A�<�:�0�0�/�#�#�#�$�0�<�<�<�<�<�<�<�0�<�I�R�U�W�U�I�=�<�0�-�0�0�0�0�0�0�0�0����� � ����������޻�������E*E7ECEJEPE\EaEfEaE\EPECEAE7E5E*E*E*E*E* 2 V 3 / > ; ) i  J #  $ t > ! H ] C , X < ;  1 <   : E  Z X 0 C P x f b R @ L � � 9  i  / � _ Z . A    �  *  S  �  )  �  �  0  �  O  �  �  �  �    �  �    �  %  I  �    I  B  �  S  u  0  <  �  �  O  �  -  �    m  a  &  �  �  \  �  j  �  �  �  �  |  �  �  	  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  
  h  �  �  ,  K  O  J  :    �  �  �  Y  
  v  �  �  -  :    Z  |  �  �  �  �  �  �  �  L    �  �  �    �  N  �  �  6  7  8  9  9  4  .  )  "        �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �      	  �  �  �  %  �  �    �  �  �        �  �  �  �  o  C    �  �  W  �  @  `  �  O  D  8  B  k  �  �  �  �  u  a  M  4    �  �  �  {  G    u  k  _  Q  B  0           �  �  �  �  v  S  A    �  j  /  $        �  �  �  �  �  �  �  �  �  �  �  �  �      �     D  ]  o  �  �  �  �  �  �  v  Q    �  �  C  �  �  4  �  �  �  �  �  �  �  �  �  �  �  �  �  �  ~  |  y  v  t  q  �  �  �  �  �  �      	  �  �  �  �  �  e  O  O  P  9    �  �  	  =  8  �  R  �  �  ]  �  V  �  �  k  �  �    )  }     (  H  `  o  k  [  D  %  �  �  �  g  :    �  p  �  �   �  �  �  �  {  ^  <    �  �  �  {  O  "  �  �  �  �  �  _  0  �  �  �  �  �  �  �  �  �  �  �  �  |  e  @    �  m  �  �  6  /  %    �  �  �  �  v  <  �  T    �  �  �  N  �  �  C  
  �  �  �  �  �  �  �  �  �  �  �  �  w  I  �  3  z  �   �  g  x  �  �  �  �  u  e  T  C  1         �  �  �  e    �  �  �    8  U  e  n  c  L  (  �  �  s  )  �  f  �  3  `  �    7  Z  w  �  �  �  �  �  �  �  �  ^  -  �  �  z    �  c  #  P  }  |  ^  G  A  <  0  #    �  �  �  �  �  �  �  �  �  Z  Y  U  Z  ]  R  A  2  @  K  F  >  ,      �  �  �  _  }  �  �  �    w  i  V  @  (    �  �  �  �  r  G    �  �  j  �  �  �  �  �       �  �  D  �  M  �  	  Z  �  
�  	�  �  �  5  2  .  )  $          �  �  �  �  �  u  B    �  �  �  [  Z  Z  Z  Z  Z  Z  [  W  N  E  =  _  �  �    8  K  _  r  �    �  �  �  s  ]  E  '  
  �  A  .  �  �  E  �  h    J  �  �  �  �  �  �  �  �  �  �  �  �  �  {  m  ^  O  @  2  #  �  �  �  �  �  �  �  ~  k  R  8    �  �  �  �  ~  ]  A  (  Q  R  N  D  6  #    �  �  �  u  <  �  �  {  5  �  �  Q  �  G  I  K  M  O  P  L  H  D  @  =  <  ;  9  8  ?  J  T  ^  h  �  �  �  �  �  �  �  �  �  �  �  �  �  �  x  m  b  W  L  A  �  �  �  �  �  �  �  �  �  �  �  p  ^  K  5     
  �  �  �  k  {  �  �  �    x  o  a  P  9    �  �  P  �  �  J   �   �  s  r  q  p  o  n  m  k  i  f  d  b  _  ]  [  X  V  T  Q  O  a  G  .  #    
  �  �  �  �  �  �  �  o  J  $  �  �  `   �  f  [  O  C  6  !    �  �  �  �  �  �  m  X  C  #     �   �    �  �  �  l  ;    �  �  �  }  V  $  �  �  �  /  �  j    �  �  �  �  �  �  x  I    
�  
d  	�  	�  	
  a  �  �  �  B  L  %    �  �  �  �  �  �  �  �  c  @  !  %    �  �  p  I  4        �  �  �  �  v  I    �     �  �  a    �  �  |  D  1  "      �  �  �  �  �  �  �  �  }  m  ]  N  @  4  '    �  �  s  a  O  8        �  �  �  }  a  G  /        �  �  �  D  �  �      
  �  �  D  �  @  �    7  
2  	  �  �  �  �  �  �  �  �  �  {  V  (  �  �  _    �  J  �  �  �  �  /  �  �  �  �  ~  q  `  I  !  �  �  �  d  :    �  �  �  e  }  E  ;  2  &    	  �  �  �  �  �  `  9    �  �  �  }  �  �  �  �  �  �  r  \  :    �  c  �  �  )  �  i  
  �  9  �    �  �  �  �  v  \  B  #    �  �  �  �  d  >    �  �  �  �  P  X  `  V  J  9  '    �  �  �  �  x  R  &  �  �  T     �  �  �  �  �  �  �  �  v  d  H  +    �  �  �  y  E    �  �  �  G    �  �  �  _  /  �  �  �  W    �  �  a  "  �  �  �  	T  	  �  �  =  �  �  ?  �  r    �  *  �  \  �  o  �  8  �