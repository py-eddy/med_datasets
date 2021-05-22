CDF       
      obs    3   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?��Q�      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       NҺ   max       P�Rh      �  x   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       �ix�   max       >+      �  D   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>p��
=q   max       @E��z�H     �      effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�fffff    max       @vS33334     �  (   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @1         max       @Q@           h  0    effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�L        max       @��           �  0h   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �]/   max       >&�y      �  14   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�μ   max       B,��      �  2    latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�x>   max       B,��      �  2�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ?/�$   max       C�Q�      �  3�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ?1uU   max       C�F%      �  4d   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �      �  50   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          =      �  5�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          3      �  6�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M��   max       PE��      �  7�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��䎊q�   max       ?�BZ�c�      �  8`   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       �ix�   max       >+      �  9,   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>���
=q   max       @E��z�H     �  9�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?ٙ����    max       @vS33334     �  A�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @'         max       @Q@           h  I�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @ʐ        max       @��           �  JP   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         F    max         F       �  K   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�IQ���   max       ?�@��4m�     �  K�         �   	         N      g         9   +   H   R      M   
      %   3      
         :      @                  `   v         3      3   "                     
         N���N���Pt��N���O�~6N�dP
�N� P�RhNd;�NEr�O�=�O�>�P4�\O�Z�O��P�N�U�NҺP�9OȺ�N�nkO��O�7�Oy��Ph7�N[@P	M�N�O?fNZYbNbk�OTm�P*l�Pg�N���N�keO�8�Nd>iP)]�Oe!N$��OmGxN�-SN�o�N-�*O@�LN_�N�NyN.5Nn~Ƚixս\)�ě���1�T���T��%   ;�`B<t�<#�
<D��<D��<T��<u<�1<ě�<ě�<ě�<���<���<���<���<���<�`B<�`B<�<�=o=o=+='�=0 �=49X=8Q�=8Q�=8Q�=<j=<j=<j=L��=L��=L��=T��=]/=ix�=y�#=�O�=���=��
=��>+����������������������������������������! %1M[gt�������gB5,!#/27<F</#`[_anz�����������zn`trpqt���������vttttt����	"-;>@B?;/"�����������������������������5N[fjg]B5)���
)-3)







���

�����������������������������*/3<HTTSQHDD<#�����|yy������������IE<0#
�������
0DSQIRQTT\aimz�����|zmaTRZY\��������������taZ��������������������dns{����{ndddddddddd�������
()-.)#
���#/<HUahmmk^UH</#ttzvz}���������ttttt��������������������"/;HPT[]QIH;/,#����������������������������������������tnmqt������ttttttttt�������
),*/;/!�����#$+'#��������������������yz����������~zyyyyyydbhit}����thdddddddd����� !&'"��pz}�������������pu���6?;-)������zu4/16BBO[g[ROKB;64444
"#&'#
#/<AHJJGA</##'0<=A<0#�����)48BDC@5)��
)25;>5) #./2/-#���);BDB6)���������	

��������������� ���������()/6;6/,)%�����������������������������������#*//00/#

###YV_admpomaYYYYYYYYYY���������������������I�U�b�n�p�o�n�b�U�I�H�B�I�I�I�I�I�I�I�I�$�0�=�C�I�V�a�^�V�I�>�=�0�$�&�$��!�$�$��)�B�O�h�|ĊĄ�`�O��������������������&�'����
���������
������������)�1�;�?�;�5�(��������������������������������v�r�k�r�~�����T�a�i�q�s�r�m�c�T�;�"�	���������!�;�H�T�.�;�A�;�7�.�"����"�+�.�.�.�.�.�.�.�.�����
�$�.�� ��������Ƨ�q�f�i�}ƇƧ�������!����������������������������������������������������������������ûԻһ��������x�l�_�T�I�S�_�l��D�D�D�EEE'E*E8E5E*EED�D�D�D�D�D�D�D߽����������н���(�:�F�A�4�����������պɺº��������κֺ����������������(�)�5�:�A�D�C�A�6�5�5�(����
���g�s���������������s�Z�A�9�0�)�,�6�C�Z�g�ѿݿ����
��������ݿۿѿ˿ѿѿѿѺ�����������������������������������"�/�H�T�\�a�\�K�"��	�����������������"�������������������������������}�w�u�{���ܻ߻��������������ܻлһܻܻܻܽ��������Ľͽ̽ƽĽ������������y�w�|������#�0�<�F�G�E�@�<�9�0�#�
�����������
��s�v�{�~�w�s�m�M�A�4�(��(�4�<�A�M�Z�f�s����������������������g�W�:�(�&�5�N�����������û˻̻û���������������������������5�B�T�`�f�g�a�5�)����	��������ʼּ������ּμʼȼʼʼʼʼʼʼʼ��tāĄĈČčĆā�t�r�h�c�[�R�O�R�[�h�q�t������������������������L�Y�d�e�n�e�b�Y�L�J�E�G�L�L�L�L�L�L�L�L�����׾���	��	��������׾ʾž�������ÓØà������á�n�a�W�]�Y�F�B�U�n�~ÈÉÓ�f�����żȼ����r�H�U�r�����{�f�X�O�T�N�fÇÓàåìíðõìäàÞÓÊÇÆÇÇÇÇ�y�������������������y�q�m�a�m�n�y�y�y�y��"�.�;�G�T�]�_�Z�T�G�"�����������������������������t�w�|�~�������m�������Ͽʿſ������y�m�T�G�;�6�9�I�T�m���
��#�/�<�<�<�;�5�/�,�#��
����������ݽ�����ݽҽнʽнؽݽݽݽݽݽݽݽݻ�!�-�8�+�!��������ݺٺ�������	��"�$�"�!���	�����׾о־׾����	�ܹ��������������ܹ׹ӹܹܹܹܺ���������������������������������������¦²¿������������������¿²§¦�������������������������x��������������ǭǡǔǏǈ�{�z�q�y�{ǈǋǔǡǭǸǱǭǭǭŭŹ��������ŹŭŪŦŭŭŭŭŭŭŭŭŭŭD|D�D�D�D�D�D{DvDoDlDoDpD|D|D|D|D|D|D|D| 3 = / ` A W = s % Z G 8 K S : G " d � G - � P & < i P > F : C & > I � ~ 6 K ^ 9 4 - X r F A S n F $ F    �       �  m  �  �  6    �  u  �  O  �  �  >  V  
  �  �  �  �  X  �    }  x  �  F  �  q  o  �  >  �  �  �  x  �  �  f  B    2  �  O  �  �    ,  ��]/��9X>\)�T��<D���ě�=��-<t�=�`B<e`B<u=�C�=aG�=�-=��`=t�=��=+<��=�%=���=C�=\)=P�`=#�
=�9X=L��=ě�=\)=q��=ix�=q��=�%>bN>&�y=T��=}�=Ƨ�=P�`=��=���=ix�=�t�=q��=��P=��=���=�9X=��=�;d>+BdnB�WB�8B�xB��B�>A�μBdBNB0B�MB"�B^hB z�B$�4A�c=B
�KB
��B(��B�~B��B�B+��A�@B 8B��Bx�B��B$�QBBْB7�B��B��BKhB�B)eBJ^B%~�B�nBW�B��B�9B#�4B�BH�BfB,��Bi�A��	BpB��B4zB��B��B�/B��A�x>B?�B�>B�B�cB"<�B�uB �XB$B6A�r�B
�+B
�|B(�rB;�B�HB�wB+��A��FB?B��B@(B��B%7OB<B=�B?WBB�BE�B)B?�B?^B%H�B>�B��B�0Ba�B#�B��BB�B4@B,��BD�A��?B!�A���B
�/A�bhA�_gA�Þ@�n<A�J�A`T B�wA�m�AJLH@��AC�Q�A*81@F��A�_�A�-A��@V A�[A�_M@�SA!LaA�WA>ZA��'@��BA�,A׹A�1@��t?�I�AS�oA�YX@��A��Ao	�A^�$@���An)�A���A*��@[AX�?/�$@��A�l�@��SBB�A�F�C��
A�k�B
`[Aԃ�A���A�@� A�OA_�B�A�u�AK �@�;9C�F%A)-�@DfpA��A���A �@T�A��A�ef@�<YA��A��A?�A��@��XA�~6AA�A�s�@�n?��tAS*aAʇ{@���AʄOAo�A^�p@���AmJA��5A*��@\;�AX��?1uU@��A��@���BF�A�H�C��	         �   	         N      h         9   +   I   R      N   
      &   3   	   
         ;      A                  a   w         4      4   "                      
                  3            &      =         %      2          #         %                  9      %                  /   =               )                                          %                  '                                 %                  3                        #   1               )                                 N���N���P�N���OVR�N�dOWULN� P0i�Nd;�NEr�N��dO�>�O�OX�O��O��LN�U�NҺO���O�N�nkO��OumXOy��PE��M��O�O�N�O)L�ND��N7�OTm�OҲP&ilN���N�keO���Nd>iP)]�Oe!N$��OmGxN�-SN��jN-�*O@�LN_�N�NyN.5Nn~�  8  �    B  +      �  @  �  �  �    (  	�  
  
  )  �  Q    �  A  $    �    �  �  (  y  M  @  �  �    b  	  S  �  	�  �  F      +  �    �  G  �ixս+<�����1��`B�T��=t�;�`B=@�<#�
<D��=49X<T��=ix�=e`B<ě�=0 �<ě�<���<�h=H�9<���<���<�h<�`B=\)=t�=Y�=o=\)=,1=8Q�=49X=�\)=��=8Q�=<j=L��=<j=L��=L��=L��=T��=]/=m�h=y�#=�O�=���=��
=��>+����������������������������������������.,07B[gt�����tg[NB5.#/27<F</#hcdnuz�����������znhtrpqt���������vttttt�� 	 ")/1330/"
������������������������5BNVZRB5)���
)-3)







���

�����������������������������*/3<HTTSQHDD<#�������������������������
#$/020+#
���RQTT\aimz�����|zmaTRediu�������������tle��������������������dns{����{ndddddddddd��������
%&*)#
����""#'./<HNSTPJH</&#""ttzvz}���������ttttt��������������������"/;HNTZ\VPHH;/$����������������������������������������qott�����tqqqqqqqqqq�������

������#$+'#��������������������yz��������zyyyyyyyyfchmtz����thffffffff����� !&'"�����������������������������*54)������4/16BBO[g[ROKB;64444
"#&'#
#/<EHIIHE></##'0<=A<0#�����)48BDC@5)��
)25;>5) #./2/-#���);BDB6)���������	

��������������� ����������()/6;6/,)%�����������������������������������#*//00/#

###YV_admpomaYYYYYYYYYY���������������������I�U�b�n�p�o�n�b�U�I�H�B�I�I�I�I�I�I�I�I�0�=�A�I�U�V�Y�V�I�=�0�'�$��$�$�0�0�0�0���6�B�N�]�^�S�6���������������������&�'����
���������
�����������!�(�+�5�8�3�(���
��������������������������������v�r�k�r�~�����/�;�H�T�_�a�c�a�_�T�C�;�/�"�����"�/�.�;�A�;�7�.�"����"�+�.�.�.�.�.�.�.�.�������������������ƢƗƔƙƝƧƳ�������!������������������������������������������������������������������������������z�x�x�x����������D�D�D�EEE'E*E8E5E*EED�D�D�D�D�D�D�D߽нݽ���� ������ݽнĽ����������Ľкֺ���������������޺ֺкʺ˺Ѻֺ����(�)�5�:�A�D�C�A�6�5�5�(����
���g�s�����������������s�g�N�D�@�@�E�Q�Z�g�ѿݿ����
��������ݿۿѿ˿ѿѿѿѺ�����������������������������������"�/�H�T�X�]�X�R�H�7�"��	�������������"�����������������������������������������ܻ߻��������������ܻлһܻܻܻܽ��������Ľͽ̽ƽĽ������������y�w�|������#�0�<�D�E�D�?�<�7�0�#��
���������
��s�v�{�~�w�s�m�M�A�4�(��(�4�<�A�M�Z�f�s�������������������g�N�/�)�5�A�Z�s�����׻����ûĻƻû����������������������������)�5�9�D�[�\�Z�P�N�5�)����	�
����)�ʼּ������ּμʼȼʼʼʼʼʼʼʼ��tāćĊċĄā�t�l�h�f�[�T�Q�S�[�]�h�t�t�������������� ���������L�Y�a�e�k�e�_�Y�L�K�G�J�L�L�L�L�L�L�L�L�����׾���	��	��������׾ʾž�������àìùþ������ûìÓ�z�n�e�]�X�]�zÇÓà�f�r�������������������y�s������f�_�W�fÇÓàåìíðõìäàÞÓÊÇÆÇÇÇÇ�y�������������������y�q�m�a�m�n�y�y�y�y��"�.�;�G�S�W�V�G�.�"��	�����������������������������t�w�|�~�������m�������Ͽʿſ������y�m�T�G�;�6�9�I�T�m���
��#�/�<�<�<�;�5�/�,�#��
����������ݽ�����ݽҽнʽнؽݽݽݽݽݽݽݽݻ�!�-�8�+�!��������ݺٺ�������	��"�$�"�!���	�����׾о־׾����	�ܹ��������������ܹٹعܹܹܹܺ���������������������������������������¦²¿������������������¿²§¦�������������������������x��������������ǭǡǔǏǈ�{�z�q�y�{ǈǋǔǡǭǸǱǭǭǭŭŹ��������ŹŭŪŦŭŭŭŭŭŭŭŭŭŭD|D�D�D�D�D�D{DvDoDlDoDpD|D|D|D|D|D|D|D| 3 5  ` E W ? s  Z G 8 K 9 - G  d � F  � P $ < j @ $ F / D + > 6 � ~ 6 G ^ 9 4 - X r C A S n F $ F    �  �  a  �  �  �  �  6  �  �  u  �  O  G  M  >  w  
  �  f  @  �  X  �         -  F  x  h  O  �  �  �  �  �     �  �  f  B    2  �  O  �  �    ,  �  F   F   F   F   F   F   F   F   F   F   F   F   F   F   F   F   F   F   F   F   F   F   F   F   F   F   F   F   F   F   F   F   F   F   F   F   F   F   F   F   F   F   F   F   F   F   F   F   F   F   F   8  2  -  '  !              �   �   �   �   �   �   �   �   �   �  �  �  �  �  �  �  �  �  �  �  �  �  �  e  D  '      a  .  �     �  `  �  �    �  �  H  �    F  d  j  
O  �  q  �  �  B  ;  3  *       	  �  �  �  �  �  �  �  �  �  v  c  P  =    #  %  (  *  *  $       �  �  �  �  b  a  H  
  �  c  �    �  �  �  �  �  �  �  �  �  �  �  �  �  w  Y  :    �  �  	  	u  	�  
>  
v  
�  
�  
�      
�  
�  
  
  	w  �    /  L  >  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  0  t  �  �        :  @  :    �  �  `    �  Y  �  '  �  u  �  �  �  �  �  �  {  k  Z  J  8  $    �  �  �  �  �  �  �  �  �  �  �  x  l  `  O  ;  '       �  �  �  �  �  �  �  �  W  {  z  o  d  X  Y  ^  `  n  �  �  �  b  $  �  ?  �  �  �    x  s  m  b  R  3     �  �  ;  �  �  =  �  G  �  H  �  �  �  �      I  n  �  �      #  (    �  �    �    s  �  �    P  �  �  	:  	|  	�  	�  	�  	�  	q  	-  �  <  �  �  �  �  �  
  	    �  �  �  �  �  �  g  A    �  �  �  �  g  >    �  	|  	�  	�  	�  

  
  	�  	�  	�  	H  �  �  K  �  r  �  M  �  �  �  )      �  �  �  �  �  �  �  f  @    �  �  �  W  '    �  �  �  �  �  o  W  ?  '    �  �  �  �  �  u  G  !  #  %  '  K  P  P  I  A  F  D  J  M  C  3  "    �  �  J  �  J  �  �  a  m  �  �  �  �             �  �  O  �  ~  �  �  0  }  �  �  �  �  �  �  u  c  G  )    �  �  h    �  �  I  �  P  A  0       �  �  �  �  v  T  /    �  �  �  �  �  �  �  �    !    �  �  �  �  �  �  �  z  N    �  �  O  �  ]  �  �        	    �  �  �  �  �  �  �  �  �  f  ;    �  �  s  Q  �  �  �    J  T  O  /    �  u    �    o  �  .  �  �  �  �  �  �  �        
  �  �  H    �  z  1  �  �  I  �  W  �  K  x  �  �  �  �  �  w  K    �  u  �  y  �  :  m  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �    "    	  �  �  �  �  l  ?    �  x  %  �  @  :  �  `    m  w  w  e  H  (    �  �  �  g  5    �  �  f  >    �  `  =  F  K  L  I  ?  )    �  �  @  �  �  q  '  �  �  ;  �  �  @  1       �  �  �  �  �  �  �  �  c  =    �  �  o    �  �  3    �  �  �  �  �  �  E  �  �    
X  	�  �  �  \  �  �  �  d  �  �  �  �  ]  �  '  Y  
R  O    �  a  
�  	�  S  �  �          0  ]  f  T  B  .      �  �  �  �  �    X  0  b  G  -    �  �  �  �  Q  %  �  �  �  �  �  �  p  8  �  �  �  	  	  	  �  �  �  �  �  �  c  -  �  �  	  _  \  @    ]  S  O  J  F  B  A  A  A  <  2  (      �  �  �  �  B   �   �  �  �  �  �  [  7    	  �  �  �  �  _  &  �  �    j  �  4  	�  	�  	�  	�  	{  	]  	:  	  �  �  .  �  3  �  �  T  �  L  .  �  �  �  �  �  �  �  �  �  �  �  r  a  N  ;  '    �  �  �  �  F  :  )    �  �  �  �  �  c  1  �  �  c    �  �  ~    1      �  �  �  �  �  �  �  �  �  p  l  q  u  z  k  X  E  2          	  �  �  �  �  |  T  (  �  �  �  2  �  |     �  +  &           	    �  �  �  �  �  �  �  o  G     �   �  �  o  ]  G  *    �  �  g  $  �  �  I    �  c    �    �    �  �  �  �  �  e  H  +    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  f  N  5    �  �  �  �  ^  &  �  �  6  �  �  G  4  !    �  �  �  �  �  �  m  S  9    �  �  �  ]  +  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �