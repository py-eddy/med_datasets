CDF       
      obs    E   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?�$�/��       �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M��   max       P��<       �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ��   max       =�G�       �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>��G�{   max       @FǮz�H     
�   �   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�p��
=    max       @vu��R     
�  +�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @&         max       @R�           �  6x   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @�_�           7   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ��`B   max       >ix�       8   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�I@   max       B5e�       9,   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�|�   max       B5@�       :@   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ?Z1   max       C��]       ;T   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ?Y71   max       C���       <h   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �       =|   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          ?       >�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          /       ?�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M��   max       P:N       @�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��PH�   max       ?�ڹ�Y�       A�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��   max       >%       B�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?\(�   max       @FǮz�H     
�  C�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�p��
=    max       @vu��R     
�  N�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @          max       @R�           �  Y�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @��           Z   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         ?4   max         ?4       [$   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�4�J�   max       ?��&��J     �  \8                        
               	      9                  !      
      U            �         ]                        5            "   	   /   '                               G               #         T      A   �   &NM�VNF�NO��OV�N�&N%�O/�OD��N��GN�'O�h�Nk��O6V.N��P��<N��OSLOa!�M�Z�N��\P,�NN�xNgvO��PD�UNu.4N5�dN�EP��N��O�ƄPU��Np�O��O�ȁN�W�N���N.8N���O͎�O�&OoڔO'g�O�U�N��/P
�pO��O<AlO
:�O�X]O 4N�&bP��N0	AN�<Nt��P�BN<ZM��O9��N)�YO\DNQ�.O��P;�N'�[O�`�O��OB%��������j��C��u�e`B�#�
��`B���
���
�D���D��%   %   %   %   :�o;D��;D��;��
;ě�;ě�;ě�<o<#�
<#�
<e`B<e`B<e`B<u<u<�o<�o<�o<��
<��
<�1<�9X<�j<ě�<ě�<ě�<ě�<���<���<���<�`B<�<��<��=C�=\)=\)=t�=�P=�P=#�
=#�
='�=,1=P�`=P�`=P�`=T��=Y�=y�#=}�=��=�G�����������������������������������������:7;BCN[gt������tg[N:((## )5?BN[ga[NB5)(����������������������������������������)058<?BDB5)������

�����aeglt�������tgaaaaaazwx{�����������{zzzz�����
%/<AB?</#
���dchlt����thdddddddd����������������������������������������IKQan������������[OI����������������������������
�������������������������5/46BHNB865555555555'%!)35BNXWNB5)''''''���)B[`bFA5)�������������������������"��������������#'����{|�����������������{`aenpz�|zuz�znla````sqtu�����tssssssssss����������������������)6BLadhTB)�������������������� �*7COVXSOC* 	)>NZ^baQ5)�������������������������������������������
/<UaihaHD/ ��-/0:<HSTPH</--------�������������������������������������#/<AHKH<9/&#/;HPT\]\YTH;/�������
#-*&#����5:HUanz�������zuaU<5��������������������
"6BO[eecOB6)
�����������������������������
#,*������spqtz�������������ys�����!().)&���)05BHNXNJB5)dgw���������������kd--./15BFNRVSNJB5----������������������������):BNSQNB)�����������������������>9;9;?BFNQSRPNJB>>>>�����������������������������'���������������������������-+,/02<@<;50--------��������������������������������������������������������������������������������--/;<HT\[YVUNHG?<3/-\a�������������{|xc\��������������������)6BHQVWRA60)��������

�����z{����������������~zĦĳĿ��ĿĳİıĦĚĔďĚģĦĦĦĦĦĦ��������������������������������������������������� ��������������üü�����6�B�O�[�h�t�y�w�t�s�i�[�O�B�@�7�0�-�4�6�����
����������
��������������������������������ﾱ���ʾ׾���������׾ʾ¾������������S�_�l�u�������x�l�f�S�F�:�6�2�8�7�:�F�S���������������������������������������Ҽ����������ɼ����������������������������y���������������~�k�`�T�G�B�:�8�K�`�o�y�L�Y�_�e�l�e�d�Y�L�G�@�F�L�L�L�L�L�L�L�L���������������������������{�u�x����������*�6�=�C�9�6�*���� ���������N�s�������	���������������s�A�&�#�*�5�N�I�V�b�o�t�{Ǉ�{�o�b�\�V�I�@�=�9�=�=�I�IŹ��������������������������ŹŲŹžŸŹ���������ĿѿʿĿ����������y�r�s�v�t�y���`�l�y�z�{�y�l�h�`�_�`�`�`�`�`�`�`�`�`�`�uƁƎƒƎƍƍƉƁ�u�o�p�o�s�u�u�u�u�u�u�
����������������¦�|�����
�6�B�E�E�B�A�6�)������ �)�-�6�6�6�6������������������üùøù��������������ƎƧƳ������������������ƮƚƎƂ�i�r�~Ǝ�[�g�s�w�z�q�g�N�)����������������)�[E�E�E�F
FFFE�E�E�E�E�E�E�E�E�E�E�E�E�ƚƧƱƳƾƳƧƚƖƗƚƚƚƚƚƚƚƚƚƚ�T�X�]�Y�T�H�G�D�H�L�T�T�T�T�T�T�T�T�T�T����������ʼּݼ߼Ӽ�����r�f�`�]�`�j����������������������������������������ҿ`�m�y�����������r�d�T�G�;�8�6�;�A�M�T�`�����	�;�a�������z�k�T�H�/�"�����������˾M�Z�`�f�s�~�s�f�Z�M�D�F�M�M�M�M�M�M�M�MÓàìòù��������ùìàÓÇÆÂÅÇÊÓ�����ȾܾԾʾ��������s�f�\�S�Z�e�s������D�EEEEEEED�D�D�D�D�D�D�D�D�D�D�D��/�<�H�U�R�H�B�<�/�&�#�#�#�&�/�/�/�/�/�/�G�O�I�G�;�;�:�0�4�;�G�G�G�G�G�G�G�G�G�G��"�.�-�'�"�����	��	����������	�"�;�T�a�m�n�h�a�Z�H�;�/�"������������(�-�7�>�;�7�3�(�������������A�N�Z�W�S�Q�U�X�P�A�5�(�����3�=�=�A�������������������������}�s�q�n�s�v����ʾ�����������׾ʾ����������������.�8�;�B�;�4�.�"��	� �������	��"�%�.�.�s����������ʾξϾɾ������Z�3�&�/�A�Z�s�нݽ���������޽Ľ������~�y�|�����Ľп������������������޿ݿݿ�(�4�A�M�M�S�M�H�C�A�<�4�-�(�!���%�(�(�A�C�Z�b�t�������s�f�Z�M�4�(�"�%�(�4�A�T�`�m�y���������}�y�m�`�X�T�I�I�T�T�T�T����������	�����������������������������I�U�n�xņň�{�j�Z�I�#���)�(�#���$�I�������������������������$�0�=�I�V�b�j�b�]�V�I�=�0�)�$��$�$�$�$����������׾Ծ׾׾��������㻞���лܼ������ܻл����x�v�w�����������������������y�n�y�~���������������������ɺֺغֺϺɺ��������������������������'�3�;�A�=�>�8�3�'����������@�D�M�T�N�M�@�5�4�3�4�?�@�@�@�@�@�@�@�@�����/�4�7�9�7�4�'�����������������������������y�s�x�y������������������������������������������������������������ùìá××ÛÍÓìù�����������	��	���������������������������r���������ǺӺպɺ��������r�_�Y�O�R�`�rD�D�D�D�D�D�D�D�D�D�D�D�D�D�DvDpDrDyD�D����������������r�f�Y�U�M�H�M�P�Y�f�r� f < I 8 g X 9 2 8 ) H  = D I J * T N b U X � 8 J b 7 b ! : E d T  b I / g K G " z 8 0 G C c . E 8 ; T 4 X g W B X V J 0  n 5 9 U F  )    �  l  �  �  g  U  |  �  �  �  �  v  �    �    �     '  �  k  �  �  �  �  �  P  ^  r  �  t  ~  �  K  �  �  �  z  �  �  9  e  y  �  �  �  	  �  Q  �  $  �  �  ]    �  �  H  1  �  E  �  �  R  :  _  �  �  ���`B���
���
�o�49X�t�;o;D��%   ;D��<�9X<#�
<t�<#�
=ix�<T��<e`B<�C�<t�<t�=�w<ě�<u=C�=��<e`B<���<���>>v�<��<�/=�;d<�9X=,1=0 �<�`B=�P<���<�`B=��-=]/=#�
=0 �=m�h=\)=�t�=�7L=L��=��=Y�=<j=D��=�+='�=@�=0 �=�G�=8Q�=8Q�=�hs=m�h=� �=aG�=��T>I�=�+>%>ix�>�+BF~B"d�B	)B��B��BbB��B#��B	�IB)[�BZ�BG~B�B�B
\�B�B̥B�BB�PBlBVB�kB��B��B�@B
&�B7>BV�B5B/��BsB�B!�[BNB�]B�=B�IBDeA�I@B� B��B�B�mB![B#�fB�Bg.B$�BgkB��B��B:JB)y�B�lB5e�B 7B @B%ҹBYcBU#B!SB,!B�B#?BZ�B��B(Bo�BcNB"E{BĤB��B�0B�kBL�B$+�B	��B)>�B?ZBAHB97B�B
EOB YB��B�dB�`B�rB��BM�BC~B��B�KB�B
?�B@XB?�B:�B/L�BCBAuB"?�B�NB�OB�B/B;�A�|�BB�B�0B�B�jB!A9B#��B��B@ZB 1BW�B�B��B��B)LDB��B5@�B@�B�vB%�zB:�BA�B!?�B,>�B��B ��BJCB@OB;�B�A��!@ �aA��Aُ�?Z1A8zAR C@���A�-�@��Ajm:?���A��mA���A�^B��A��Ar��A~B%�A��oA�-oA���B�A��XC��]B�RA�k?@�j�A�J�Ai~�A�ҚA?�uA���AH�BC�T�A�[yAc�A��A���A��zA���AG�:AR�{A^: AD�A%+�A��6A9A=a�Aj�`A���A�{�@Y��B
��AU��@�NAp�@/�3?�?�@��'@�`�A�:A��A� PA�W@�$C��@@��A�Y!@S@A��AـH?Y71A �AR�@���A��4@��Aj�P?�g�A��eA�m�A�u$B�xA���Ar�A �B@iA��AׁA�u�BW�A�p�C���B A��@� NA�u AjA���A?�`A��AE��C�W�A�`Ab��A�_�A��[A��A��gAG�AR�A]�AF�pA'�A��A9O^A=M�Aj�_A�}xA@[6B6-AV�k@��Aq �@,L�?�\@��X@�� A��A�G�A�~�A�wb@`�C���@�8/                              	         
      :                  "      
      U            �         ^            	            6            "   
   /   (                               G               $         U      A   �   &                                             ?                  /         !   -            %         3         )               !                  +   #         !         )            +                        -      !                                                   /                  +         !   #                              %                                 %   #                  )                                    #      !      NM�VNF�NOc%mN��ZN�&N%�O�~OD��N��GN��LOz'�N&��O6V.N��P:NNś1OSLN�r1M�Z�N��\P�N`�zNgvO�3�O� �Nu.4N5�dN�EO�B�N�$�O�ƄO��Np�OKO�#N�W�N���N.8N���OrܨO�&OoڔO
D~O�<�N��/O��SO��Ob�N�*�O�;�O 4N�&bP��N0	AN�C�Nt��O�s1N<ZM��N��N)�YOJ	NQ�.N�oO�/NN'�[O��OD7dOB%  u  '  �  -  �  �  *  �  7  A  O  W    j  �  �    \  t  �  �  H  �  �  �  �  �  Z  �  +  M  
C    �       �  <  '    1  �  �  �  �  M  o  �  �  �    f    �  �  4  	  ^  ?  c  L  )    �  	�  �  	   j  	L����������T���u�e`B�o��`B���
��o:�o��o%   %   <�o:�o:�o;�`B;D��;��
<#�
<o;ě�<49X=\)<#�
<e`B<e`B=���<�C�<u=e`B<�o<�t�<�9X<��
<�1<�9X<�j=��<ě�<ě�<���<�`B<���=+<�`B=o=o=+=C�=\)=\)=t�=��=�P=��=#�
='�=H�9=P�`=Y�=P�`=q��=�\)=y�#=�hs>%=�G�����������������������������������������AABHN[gt}�����tg[NGA1.+,-5BBNV[\[WNBB511����������������������������������������),56:;>50) ������

�����aeglt�������tgaaaaaayy{�����������|{yyyy����
 /6<>?<8/#
��ghhqty����thgggggggg����������������������������������������ST^ht������������g[S����������������������������
�������������������������5/46BHNB865555555555'%!)35BNXWNB5)''''''����)5N[_C<5)������������������������"��������������#������������������������`aenpz�|zuz�znla````sqtu�����tssssssssss��������������������	
)6>BLNOJB<6)	�������������������� �*7COVXSOC* %)5BEIOPNJB5)�������������������������������������������
,<UaeeaH</# �-/0:<HSTPH</--------�������������������������������������#/<AHKH<9/&#,'$"#%-/;HMTWVSMH;/,�������
#-*&#����5:HUanz�������zuaU<5��������������������)6BO[bc`OB6)��������������������������
! 
�������spqtz�������������ys�����&'�����!)-5BDNTNHB5)skq{���������������s--./15BFNRVSNJB5----������������������������):BNSQNB)�����������������������?;<:<ABCNPRRONGB????������������������������������������������������������������-+,/02<@<;50--------��������������������������������������������������������������������������������//4<HOUWVUQH<80/////lhhq���������������l��������������������)6BJQQI>6)��������


�����z{����������������~zĦĳĿ��ĿĳİıĦĚĔďĚģĦĦĦĦĦĦ������������������������������������������������������������������������������6�B�O�[�h�n�h�h�c�[�P�O�L�B�<�6�6�4�6�6�����
����������
��������������������������������ﾾ�ʾ׾��������׾ʾ����������������S�_�l�u�������x�l�f�S�F�:�6�2�8�7�:�F�S���������������������������������������Ҽ��������Ǽ������������������������������m�y�����������{�m�e�`�T�K�G�A�?�G�Q�`�m�L�R�Y�e�h�e�`�Y�L�I�C�L�L�L�L�L�L�L�L�L���������������������������{�u�x����������*�6�=�C�9�6�*���� ���������Z�s�������������������s�Z�A�8�2�1�?�B�Z�I�V�b�o�r�{�|�{�o�b�W�V�I�B�=�:�=�?�I�IŹ��������������������������ŹŲŹžŸŹ�������������¿����������}�z�������������`�l�y�z�{�y�l�h�`�_�`�`�`�`�`�`�`�`�`�`�uƁƎƒƎƍƍƉƁ�u�o�p�o�s�u�u�u�u�u�u�����
��	������������¦¦¿�����6�B�B�B�B�@�6�)��!�)�1�6�6�6�6�6�6�6�6������������������üùøù��������������ƎƧƳ������������������ƳƚƊ�~�p�yƁƎ�)�5�B�N�[�b�i�g�\�?�)��������������)E�E�E�F
FFFE�E�E�E�E�E�E�E�E�E�E�E�E�ƚƧƱƳƾƳƧƚƖƗƚƚƚƚƚƚƚƚƚƚ�T�X�]�Y�T�H�G�D�H�L�T�T�T�T�T�T�T�T�T�T�����������ȼǼ�����������v�r�o�n�q�~�����������������������������������������ҿ`�m�y�����������r�d�T�G�;�8�6�;�A�M�T�`���	��/�6�H�N�R�L�H�;�/�"��	����������M�Z�`�f�s�~�s�f�Z�M�D�F�M�M�M�M�M�M�M�MÓàéìù������ÿùìàÓÇÇÄÇÇÑÓ�����þ־˾���������f�`�Z�_�g�k�s������D�EEEEEEED�D�D�D�D�D�D�D�D�D�D�D��/�<�H�U�R�H�B�<�/�&�#�#�#�&�/�/�/�/�/�/�G�O�I�G�;�;�:�0�4�;�G�G�G�G�G�G�G�G�G�G��"�.�-�'�"�����	��	��������	��"�/�H�T�a�b�d�Z�T�H�;�/�"�����	����(�-�7�>�;�7�3�(�������������A�N�Z�W�S�Q�U�X�P�A�5�(�����3�=�=�A������������������������s�s�o�s�x�������ʾ׾�����������׾ʾ������������.�8�;�B�;�4�.�"��	� �������	��"�%�.�.��������Ǿƾ���������f�Q�:�0�7�M�Z�s��нݽ���������޽Ľ������~�y�|�����Ľп������������������߿�������(�4�A�E�M�Q�M�F�B�A�9�4�0�(�"� � �'�(�(�4�A�M�Z�_�f�s�}����s�f�Z�M�A�4�+�&�'�4�T�`�m�y���������}�y�m�`�X�T�I�I�T�T�T�T����������	�����������������������������I�U�n�xņň�{�j�Z�I�#���)�(�#���$�I�������������������������$�0�=�I�V�b�e�b�Z�V�I�=�0�-�$��$�$�$�$����������׾Ծ׾׾��������㻪�ûлܻ����ֻлû����������������������������������y�n�y�~���������������������ɺֺغֺϺɺ��������������������������'�3�5�;�8�7�3�'�����	������@�D�M�T�N�M�@�5�4�3�4�?�@�@�@�@�@�@�@�@������&�4�5�7�6�4�'����������������������������y�s�x�y���������������������	���������������������������������������������ùìååèããíù���������	��	���������������������������r�~�����������ʺȺ��������r�e�W�Z�e�l�rD�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D|D~D�D�D����������������r�f�Y�U�M�H�M�P�Y�f�r� f < J - g X ; 2 8 , H $ = D F B * ? N b U C � 3 G b 7 b  ; E H T  d I / g K ; " z * . G K c , F 3 ; T 4 X g W ' X V E 0  n  8 U E  )    �  l  �    g  U  1  �  �  �    B  �    V  �  �    '  �  �  x  �  �  0  �  P  ^    �  t  T  �  -  ^  �  �  z  �  �  9  e  -  G  �  �  	  a  !  �  $  �  �  ]  �  �    H  1  '  E  �  �  �  Z  _  W  �  �  ?4  ?4  ?4  ?4  ?4  ?4  ?4  ?4  ?4  ?4  ?4  ?4  ?4  ?4  ?4  ?4  ?4  ?4  ?4  ?4  ?4  ?4  ?4  ?4  ?4  ?4  ?4  ?4  ?4  ?4  ?4  ?4  ?4  ?4  ?4  ?4  ?4  ?4  ?4  ?4  ?4  ?4  ?4  ?4  ?4  ?4  ?4  ?4  ?4  ?4  ?4  ?4  ?4  ?4  ?4  ?4  ?4  ?4  ?4  ?4  ?4  ?4  ?4  ?4  ?4  ?4  ?4  ?4  ?4  u  {  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �    '        �  �  �  �  �  �  �  �  �  t  O  +    �  �  M  �  �  �  �  �  �  �  �  �  �  �  �  �  t  ]  8  
  �  �  2    
      (  -  (        �  �  �  �  �  �  o    |   �  �  �  �  �  �  ~  y  t  o  k  e  ^  W  P  I  =  0  "      �  �  �  �  y  l  _  Q  C  4  &      �  �  �  �    c  G    #  (  *  '  #          �  �  �  �  �  �  �  �  �  �  �  �  |  s  j  ^  R  E  7  (      �  �  �  �  �  �  �  �  7  /  &      
     �  �  �  �  �  �  �  �  �  �  �  �  �  A  A  A  >  :  4  *      �  �  �  �  �  k  B     �   �   �    :  I  O  K  >  &    �  �  �  f  K  %  �  �  \    �  �  7  F  P  W  R  F  5  %    �  �  w  >  �  �  o  )  �  �  G          �  �  �  �  �  �  �  �  �  �  �  �  v  Y  0    j  d  ^  V  M  @  3  "    �  �  �  �  �  �  �  �  �  �  �  �    t  �  �  �  �  �  �  �  m  &  �  �  7  �  k  �  �   �  �  �  �  �  �  �  �  �  p  S  5    �  �  �  �  f  2    �          �  �  �  �  �  �  �  �  t  ^  E  +    �  �  �  �      (  3  O  Y  T  N  F  <  0  !    �  �  �  �  Y  �  t  b  P  ?  )    �  �  �  v  Q  7      �  �  �  �  z  ^  �  �  �  �  �  �  �  �  �  �  �  �  �  �  v  ^  C  '     �  �  �  �  �  �  �  �  �  p  W  *  �  �  �  =  �  �  o  
  �  F  #  E  ^  s  �  �  �  �  �  �  �  �  �  �  �  �  �      �  �  �  s  e  ]  U  H  7    �  �  �  R    �  a    �  t  �  �  �  �  �  �  �  �  �  z  l  ^  H  )  �  �  d  �  �     �  $  h  �  �  �  �  �  �  �  �  t  1  �  R  �  -  M  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  |  q  f  Z  N  @  2  $    �  �  �  {  4   �   �  Z  U  P  K  D  7  *      �  �  �  �  �  �  l  O  0     �  �  �  �    k  �  �  �  �  �  Q  �  .  T  D  �  Z  
~  _  �       *  )  &         �  �  �  �  �  t  c  >     �  j    M  @  2  "    �  �  �  �  �  �  �  �  u  U  O  O  A  9  4  "  �  �  	  	m  	�  
  
8  
C  
=  
/  
  	�  	�  	?  �  �  �  d  �    �  �  �  �  �  �  �  �  �  z  i  Y  I  :  +    
  �  �  �  �  �  �  �  �  s  N  %  �  �  �  J  
  �  �  T    �  �              �  �  �  �  �  �  �  s  @  �  �  f        �  �  �  �  �  �  �  o  [  E  /      �  �  �  �  L    �  �  �  �  �  �  �  �  �  �  j  E    �  �  �  L  
  �  e  <  6  1  +  &          �  �  �  �  �  �  �  �  �  }  i  '         �  �  �  �  �  �  �  �  {  l  ]  N  L  M  N  O  o  �  �  �            �  �  �  �  |  9  �    I    �  1  )  !        �  �  �  �  ~  U  )  �  �  s    �  $  &  �  �  �    V  A  y  _  N  5    �  �  �  V  "  �  �  �  �  �  �  �  �  �  �  �  |  c  @    �  �  �  I  �  �  .  �  0  �  �  �  �  �  �  �  �  �  x  S  &  �  �  y  4  �  z  �   �  �  �  �  |  w  t  r  g  X  G  4      �  �  �  �  \  -   �    *  H  M  E  7  &    �  �  �  �  �  �  j  $  �  <  c  W  o  g  [  N  N  h  Z  F  *    �  �  {  0  �  �  �  8  -   �  �  �  �  �  �  �  �  w  g  U  @  "  �  �  �  7  �  �  '  �  �  �  �  �  �  �  �  �  �  {  n  `  P  @  0         �  �  �  �  �  �  �  �  �  �  ~  g  N  0    �  �  �  ]  1  �  $      �  �  �  �  �  �  n  Q  2    �  �  �  i     �  o    f  c  _  Z  N  >  +    	  �  �  �  �  �  �  �  �    T  �        �  �  �  �  �  }  _  :  
  �  �  f  '  �  �  >    �  �  �  �  m  V  ?  (    �  �  �  �  �  �  �  �  �  {  p  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  _  8    4  '      �  �  �  �  �  �  n  J  &  �  �  �  p  !  �  �  �  <    �  �  �  	  	  	
  �  u    |  �  ^  �  F  G  �    ^  ^  ^  ^  \  X  S  O  H  @  7  .    �  �  �  �    a  C  ?  3  '        �  �  �  �  �  �  �  h  L  3       �   �  �  &  B  V  a  `  Y  N  =  #  �  �  �  U  %  �  �  Q  �  �  L  I  G  C  =  7  /  %        �  �  �  �  �  �  �  �  �  $  (      �  �  �  ~  P     �  �  �  f  >    �  �  �  �             �  �  �  �  �  �  �  �  �  �  �  |  Z  9    �  <  �  �  �  �  �  �  r  S  (  �  �  �  Q    �  �  C  �  	  	Q  	t  	�  	�  	  	x  	f  	a  	U  	A  	  �  r    �  �      �  �  �  �    v  g  Y  K  6    �  �  �  �  u  S  .  	   �   �  �  �  	  	  	  �  �  �  �  �  ]  "  �  �    X  �  _    �  s  �  N  �    A  d  e  V  1  �  �      �  q    x  \  	O  	L  �  �  �  ~  ]  /  �  �  �  A  �  �  )  �  �  $  V  :   �