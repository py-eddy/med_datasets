CDF       
      obs    C   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?����E�       �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M��   max       P�`�       �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       �ȴ9   max       <�1       �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�33333   max       @F��Q�     
x   �   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��\(�    max       @vW\(�     
x  +H   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @"         max       @P            �  5�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @̶        max       @�(`           6H   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �n�   max       ;�`B       7T   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�L�   max       B4��       8`   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A��4   max       B4�,       9l   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >��   max       C��       :x   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >z�`   max       C���       ;�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          a       <�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          G       =�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          E       >�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M��   max       P���       ?�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��{���n   max       ?�C,�zy       @�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       �ȴ9   max       <���       A�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�33333   max       @F�Q��     
x  B�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��z�G�    max       @vW\(�     
x  MP   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @!         max       @P            �  W�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @̶        max       @���           XP   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         D-   max         D-       Y\   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?���u��"   max       ?��PH�     �  Zh                           '                            	      3         7   1      $         	   
               !   ,   5      ,   
      '      	      a      )                  /      	   F      /         
   .         
   O(^NXNN�8�Ow�N��RN���N�{O0eP�vO�>O��N��aO�o�N�Nha�N9h�O��MN�6N!ƦP� kPg;N��P�`�Pz*DN���O��N�26O`��N�i�N�]�N�ȓO (O�~�Nq�bO\��P^�PN��O..O��	N%}N�Q^O�-O���O cN�9�P�#�NFv�OZ�O��N�̵O�%O
cWO��<O���O��fN��P3�VN�&�O�a�OcabM��O��O�J�NrѶN��bN��N�
<�1<T��<T��;�o;D��%   ��o�o�o��o��o�ě���`B�o�#�
�#�
�49X�T���e`B�u�u�u�u�u��C���C���t���t��ě���`B���+�+�+�C��\)�\)�\)�\)�t����#�
�#�
�'0 Ž@��D���D���H�9�T���Y��]/�e`B�e`B�q���q���u�y�#��%�����O߽�\)��hs���P���
�� Žȴ9#/<?AE<:;6/#()36BFO[OHB61)((((((�������������������� #*/1<<<>GFC</#" FNR[`gkjg][NHCFFFFFFghjqt|�����tphgggggg������������������������������������$5N[t~}ytgaNB)LPQ\ht����������phQL"6Od\TYWYTOHC6*S[eht{����tstytqeUSpt��������������{�xp�����������������������������������������k|�������������ztjgk"$/;=DB;/" 5<=HHKH@<95255555555,;HTz��������mPL6"%,z���������������{orzU[gt���������vtig[UU�����#<W{���pI0�������������������imz~�������}�zyyvrkiw����������������}ww����������������������)5864)�����xzz���������zwxxxxxx
!#+-*%#
����������������������������������������5Bgt�������tWNB5%����������������')6BO\]]ZWQOBA62)%#'n������
���������zhn������������������)6BFJJEB6)0<IOR[ev|~|th[B)NOR[hnjh[ONNNNNNNNNNMO[_bb`\[OOLKKMMMMMM
#'/CU\][UH<#	
).6BMRSROMB69BN[gtzztrng][PNHB?9����� ����������EHaz����������zneVFE7;=HPSMH;87777777777���������������������������� ����������������������� ������������������}~~}������������������������$5A:)�������z�����������������{z�������������������������$�������55;BN[[ghgf[SNB54255���
#<HU]_UNH<#
���U[]gt��������tgb[YSUttu������ttttttttttt��������������������5B[iiefog[B5)

#'/35/#






���������}wy}��� 
	���������BIQUbbddcbUPMIFDBBBB�Y�N�H�@�A�B�N�[�b�g�t�t�g�Y�/�,�#�"�"�#�)�/�4�<�?�?�<�2�/�/�/�/�/�/�����������������žȾ��������������������/�,�#���!�#�/�<�H�U�a�f�n�k�a�U�H�<�/���������������ĿɿѿտݿѿĿ������������Ŀ��������������Ŀѿٿ׿ѿſĿĿĿĿĿ�¥¦®²³²²¦�s�m�g�Z�Z�Y�\�e�g�s�v�����������������s�5�(�
����(�5�A�N�U�_�g�x�z�x�s�g�Z�5��r�f�M�B�:�6�>�M�Y�f�r���������������	���پ������ʾ׾��	��"�2�=�?�;�.�"�	�нɽĽ½ɽнݽ�������	�������ݽллƻ˻û����������ûܻ���'�(���ܻо������������������������ľƾ¾����������$�#�$�$�$�$�$�0�9�=�>�>�>�=�8�0�$�$�$�$�0�/�,�0�=�I�O�I�F�=�0�0�0�0�0�0�0�0�0�0�����������������������$�+�+�)�$���������"�#�0�<�B�B�>�<�0�#������ǈǆǈǔǖǡǥǭǴǭǡǔǈǈǈǈǈǈǈǈ����������������/�;�Q�E�T�_�T�;�	������� ��������������6�O�e�uƄƎƚƋ�h�C�� ����������������� ��������	���������|�s�h�N�4�2�g�������� ���	�ɿ��y�`�S�M�T�m�����Ŀݿ�������$��ݿ�ìàÚàìù������������������������ùì�)��������������B�O�]�^�[�^�Z�B�6�)������#�/�<�C�H�U�Q�H�<�/�#����ŠŞŔŏŎőŔŠŭŹ��������������ŹŭŠ��������������������������������������$��"�$�$�0�=�I�V�X�V�U�I�@�=�0�$�$�$�$ƳƱƮƳƸ������������������ƳƳƳƳƳƳ���}�y�v�y�{�����������ĿƿοĿ�������������������	�"�4�:�5�"�����	��àÜÓÇÅÇÈÒÓØàêìâàààààà�s�q�o�m�k�s���������������������������s���� �)�/�@�H�b�������������z�a�>���-�!�������!�F�V�l�����������������F�-�_�W�S�L�H�G�O�S�_�l�x�|��|�|�{�x�l�_�_�5�"�	�������������	��"�/�>�I�P�U�P�H�5�Ϲ̹ù����ùϹӹ۹ѹϹϹϹϹϹϹϹϹϹϿ	���	��"�.�;�;�;�;�.�"��	�	�	�	�	�	�z�\�h�s�|�����������������������������z�s�i�m�s�y���������ʾؾ׾վ;���������s����������������� ����������������	�� �	����"�.�1�4�/�.�"���������������ù���'�@�Y�f�q�l�R�@�'���ù����������$�#�������������������(�5�A�N�R�P�N�J�A�5�(���������{�y����������������������������������ټݼ������������������������߽������(�4�:�8�4�)��������������*�6�C�G�D�O�P�O�C�6�*���������$�0�=�V�d�j�o�b�I�=�0�$��
����čćăčĘğĦĮĳĿ������������ĳĦĚč�0�.�#���
�������0�<�Q�b�o�x�k�F�<�0ŔŉŔŜŠŭŹż��������������ŹŭŠŔŔ�r�Y�D�A�H�r�����ͼּ��ڼμȼ¼������r���	���	�	����"�+�/�1�9�0�/�"����ݿֿϿȿƿ����Ŀѿݿ����0�:�1����ĳĪĦĜĔĔĕĚĦĳĵ��������������Ŀĳ¿¼²®²²¿����¿¿¿¿¿¿¿¿¿¿¿���� ��'�4�M�\�f�j�j�f�^�Y�@�4�'���Ó�z�a�P�L�L�U�a�zÓàéáÝÞâäàØÓECEBE7E1E2E7EAECEPEPEREQEPEDECECECECECEC�����#�/�<�H�K�K�H�<�/�#�������G�<�G�S�`�e�l�y�������������y�l�`�S�G�G�нǽнӽݽݽ�����	������ݽннн� G Z 9 D + . E , 1 , m y T , b ; " @ H O 1 w n D z ( D G 0 ? 1 R j 9 2 H 2 J 7 < ^ \ R D 4 O 5   Z C W a k 3 k A 5 L ? G � G M \ } Z    �  �  �  p  �  �  �  B  �  q  �  W  
    �  H  2  �  Y    �  D  �    �  �    �  �  �  �  �  v  �  �  �  �  �  *  M  �  �  S  a  �  �  N  �  S  �  b  ]  �  �  �  -  W    q  �  &  �    �  �  C  Ǻ�o;�`B;ě��D�����
�ě��u�e`B�#�
�C���1�#�
��㼓t��u��C���w��1��1��+�<j��t���\)��%���ͽT���t���P�+���t��49X��o�#�
��+������9X��%�����8Q�'��w��+�L�ͽaG��n��aG���E���t��������u������񪽰 Ž�7L�����-��;d��{���P�������vɽ�Q�\��S�B�~B��B"U#BH�B�_BzyB�yB>B�qB~�B0<B��B �B4��Bp�B�7B wA�L�B��A�0SB�zB	��B%��B+��A��XB��B��B�TB YBHWBBhB��B	�SB��B2�B_>B}eBp�B��BN�B�Bu�B��B��BmB�/A�U.B�B��B.,B�[B
��B,B>�B�B�jB�sB(B��B	��B�{B*�Bg~B�TB
HB`�B'!�B�\B�B"@B�;BD�Bw�BƨB7�B�(B�|B0H	B�8B 2�B4�,Bv�B�B R	A��4B�A��B{ B
�B%��B,�A��YB��B�BņB @iB�;B@�B��B	��B��B?BqBB�2B?�BD,B�1B��BGB	@B�[B��A�w�B;qB@iB.��B�=B)�B��BN{BE�B�@B�B �B>B	M�B��B*@nB BDFB
�3BrmB'@ A���A�R'AK�hA��Axe-AxP�A���A��5A��x@ۣ}AZ�A-Y{@���AK�B
Y"B
�gB|_A�(�B�cA�~�B i}A.�A�V�Aw}�A�!A��"A�A��A�fB
�GB�As'A[�A�IA�sA��}@�LP@���A�R>��A^�A�x�AJ�A���A]��?Ku=A��<A���A��Ax�A3n0A�]ZB
lgA�RA�]�A�,J@��cA��A��YA� �A�,@�{=A�pC��A�fAs6A,�iA��$A8AI�A��Ax�JAx�GA��A�qA���@��AZ	�A*հ@�BAL��B
*B
��B��A�~�BIRA��3B �&A/O�A���Av��A���Aׇ A�lcA���A��B
ɸB�CArQ�A[ sA�k�A�v�A���@�	@�^A��>z�`A]A��AJ �A���A]�?O�RA���A�v�A��)A50A4��A��&B
��A��A�SA�p�@䵵A��(A�y<A�sA�{@Ѕ�AȄC���A��A4kA-+�               	            '                            	      4          8   1   	   $         
                   !   -   6      -   
      '      
      a      *                  0      	   F      /         
   /         
                              %   !   #      +                     ;   3      G   ;      %                     !         7   /      '         )            5                     !      %      1      '            %                                       #                                 %   3      E   ;                                    1   /      #         '            5                     !            +      !                        O}�N&TN��Ow�N��#N���N�{N�O��Nb6�Og��N��aOZxN��bNha�N9h�OJ-�N�6N!ƦP�Pg;N��P���Pr�_N�|�OT�sN�26O�N�i�N�]�NFO (O&!�Nq�bO.��P85PI��N�Q�Oڛ1N%}N�Q^O�N�O3�HO cN�9�Pv�mNFv�OK�wO��N�̵O&5O
cWO��<O~��Ot��N��P$'yN�&�O��O*��M��O��O��NWN��bN��N�
  i  �  �  �  �  >  A  �  �  }    �  �  A  �  �  �  +    �  �  K  �  �  `  a    ^  �  �  7  ~  �  �    f  �  {  �    �  5  �  &  �  	K  �  �  .    :  �  "  �  �  �  	�  a  T    �    �  �  [  �  \<���<D��<D��;�o;o%   ��o�D���#�
��j�ě��ě���C��t��#�
�#�
��C��T���e`B�C��u�u��o��o��t�����t���9X�ě���`B���+�<j�+����w�t��0 Žt��t����,1�8Q�'0 ŽH�9�D���L�ͽH�9�T���]/�]/�e`B�q����+�q����o�y�#��\)��O߽�O߽�\)���P�������
�� Žȴ9#/<=<<995/#))6BOTOFB63)))))))))�������������������� #*/1<<<>GFC</#" HNW[]gjhg[YNJEHHHHHHghjqt|�����tphgggggg������������������������������������#)5BN[gswwtc[NB)hhty�������tihhhhhhh$6<COUTVTQLEA6*S[eht{����tstytqeUS�������������������������������������������������������������lqyz������������zpml"$/;=DB;/" 5<=HHKH@<952555555557;HTam����zmaOD<217z���������������{orzU[gt���������vtig[UU����#<Y{���~oI0�������������
 �������jmtz|������|zzwsmljj�������������������������������������������()/.)$����xzz���������zwxxxxxx
!#+-*%#
����������������������������������������Q[gpt���������tkgYOQ����������������()67BOZ[XUROB65,)'%(u�����������������u��������� ���������#)6?BEEB:61) 1=JOQ[dv|~{th[B) NOR[hnjh[ONNNNNNNNNNMO[_bb`\[OOLKKMMMMMM
#%/BU[YUH</#

)36BIMNFB?6)9BN[gtzztrng][PNHB?9����� ����������HUaz���������zneWJFH7;=HPSMH;87777777777���������������������������� �������������������������������������������}~~}������������������������ 74)����������������������������������������������������"��������55;BN[[ghgf[SNB54255�
#/HJHIC</#
������Y[`got�������tg\[VYttu������ttttttttttt��������������������)5BN[dfaa[NB5)#'/0/-#
���������}wy}��� 
	���������BIQUbbddcbUPMIFDBBBB�g�\�[�N�J�C�K�N�Y�[�g�t�t�g�g�/�/�#�#�-�/�1�<�>�>�<�/�/�/�/�/�/�/�/�/�����������������¾ľ��������������������/�,�#���!�#�/�<�H�U�a�f�n�k�a�U�H�<�/���������������ĿǿпѿٿѿĿ������������Ŀ��������������Ŀѿٿ׿ѿſĿĿĿĿĿ�¥¦®²³²²¦�s�p�g�[�Z�Z�Z�^�g�g�s�s���������������s�Z�N�5�*������(�5�A�N�Z�g�p�t�q�g�Z�Y�Y�M�M�L�M�U�Y�e�f�n�k�f�\�Y�Y�Y�Y�Y�Y��	���ܾ׾ξ̾׾����	��"�.�9�<�7�.��нɽĽ½ɽнݽ�������	�������ݽлл˻û����������ûлܻ����� ����ܻо����������������������¾ž��������������$�#�$�$�$�$�$�0�9�=�>�>�>�=�8�0�$�$�$�$�0�/�,�0�=�I�O�I�F�=�0�0�0�0�0�0�0�0�0�0�����������������������$�&�'�$��������"�#�0�<�B�B�>�<�0�#������ǈǆǈǔǖǡǥǭǴǭǡǔǈǈǈǈǈǈǈǈ���������������������"�5�9�9�"�	��������� ��������������6�O�e�uƄƎƚƋ�h�C�� ����������������� ����������������}�s�i�N�7�?�g�}�����������
��̿��y�`�T�N�`�m�����Ŀݿ�����������ìãàÝàìðù����������������ûùìì�)�������
���)�B�O�T�S�T�O�B�6�)������#�/�<�C�H�U�Q�H�<�/�#����ŭŤŠŜŖŒŔŔŠŭŲŹ����������ſŹŭ��������������������������������������$��"�$�$�0�=�I�V�X�V�U�I�@�=�0�$�$�$�$��ƻƳƯƳƻ�������������������������������}�y�v�y�{�����������ĿƿοĿ�����������������������	���"�(�&�"� ��	��àÜÓÇÅÇÈÒÓØàêìâàààààà���v�s�s�q�t�����������������������������"����%�+�7�H�a�m�������������z�a�H�"�-�!�������!�F�Y�l�����������������F�-�_�Z�S�P�M�S�W�_�l�s�x�z�x�v�v�l�_�_�_�_�"�	�������������	��"�/�=�I�O�T�O�H�/�"�Ϲ̹ù����ùϹӹ۹ѹϹϹϹϹϹϹϹϹϹϿ	���	��"�.�;�;�;�;�.�"��	�	�	�	�	�	��c�m�s�~�������������������������������{�w������������ʾ̾ʾƾ����������������������������� ����������������	�� �	����"�.�1�4�/�.�"���������������ƹ���'�@�Y�o�k�Q�@�'���Ϲ������������$�#�������������������(�5�A�L�N�Q�O�N�H�A�5�(��������{�y����������������������������������ټݼ�����������������������������(�4�8�6�4�(�&����������������*�6�C�G�D�O�P�O�C�6�*���������$�0�=�V�d�j�o�b�I�=�0�$��
����čĉĄĎęġĦĳĿ������������ĿĳĦĚč�I�<�0�#�����"�0�<�I�S�U�b�c�e�b�V�IŔŉŔŜŠŭŹż��������������ŹŭŠŔŔ�Y�E�B�J�Y�r�������üּռʼż��������r�Y���	���	�	����"�+�/�1�9�0�/�"���ݿӿͿ̿ؿݿ������*�4�+��������ĳİĦğĚėėĚğĦĪĳĿ����������Ŀĳ¿¼²®²²¿����¿¿¿¿¿¿¿¿¿¿¿���� ��'�4�M�\�f�j�j�f�^�Y�@�4�'����z�c�Y�Q�M�M�O�U�aÇÓÝÞÛÜáâÝÓ�zE7E2E3E7ECECEHEPEREQEPECE7E7E7E7E7E7E7E7�����#�/�<�H�K�K�H�<�/�#�������G�<�G�S�`�e�l�y�������������y�l�`�S�G�G�нǽнӽݽݽ�����	������ݽннн� D d ; D 1 . E % + L \ y F ) b ;   @ H N 1 w m A c / D = 0 ? 1 R 9 9 # @ 5 < 5 < ^ Y B D 4 N 5   Z E W a g 4 k = 5 ( A G � B D \ } Z    2  �  �  p  �  �  �    �  �    W  �  �  �  H  �  �  Y  �  �  D  �      �    ]  �  �  V  �  f  �  m  P  f  �    M  �  �  �  a  �  Y  N  �  S  �  2  ]  �  U  �  -  �    �  �  &  �  �  {  �  C  �  D-  D-  D-  D-  D-  D-  D-  D-  D-  D-  D-  D-  D-  D-  D-  D-  D-  D-  D-  D-  D-  D-  D-  D-  D-  D-  D-  D-  D-  D-  D-  D-  D-  D-  D-  D-  D-  D-  D-  D-  D-  D-  D-  D-  D-  D-  D-  D-  D-  D-  D-  D-  D-  D-  D-  D-  D-  D-  D-  D-  D-  D-  D-  D-  D-  D-  D-  B  a  g  ]  O  B  6  1  /  6  N  T  I  4    �  �  Q  �  B  �  �  �  �  �  �  �  �  �  �  �  �  �  h  J  ,    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  k  T  ?  +     �   �   �  �  �  �  �  x  h  d  �  �  �  t  a  H    �  ~  
  �  4   �  u  z      }  y  o  e  W  H  9  )    	  �       )      >  ;  8  6  ,         �  �  �  �  �  �  �    h  N  3    A  3  #      �  �  �  �  �  _  8    �  �  �  `  K  `  �  �  �  �  �  �  w  [  ?  #    �  �  �    X  /    �  �  5  d  �  �  �  �  �  �  �  _  <    �  �  �  {  R  #  �  :   �  )      �  �      -  6  J  f  t  |  q  Z  :    �  v  �  �  �            �  �  �  �  �  �  m  R  8    �  �  O  �  �  �  �  �  �  �  �  s  e  Y  P  G  ?  6  -  %        "  +  8  G  b  ~  �  �  �  �  �  h  V  P  6  �  �  y  E  �  8  =  A  @  >  >  >  =  :  4  *      �  �  �  r  D     �  �  �  �  �  �  �  �  �  �  y  [  =    �  �  �  �  J     �  �  �  �  �  �  �  �  �  �  f  D  "  �  �  �  }  R  $   �   �  �  �  �  �  �  �  �  �  �  �  �  �  ]  -  �  �  n  "  �  T  +  !      
    �  �  �  �  �  �  �  f  C    �  �  �  i    	  �  �  �  �  �  �  �  a  A     �  �  �  �  f  ;     �  �  �  4  S  k  �  �  �  �  �  �  d  &  �  e  �  ;  �  b  �  �  �  �  �  �  �  �  |  i  P  4    �  �  n  C  +    �  �  K  L  M  N  O  P  Q  N  I  D  ?  :  5  0  ,  '  #        �  �  �  �  �  �  �  �  w  F    �  r    �  %  �  q  �   �  �  �  �  �  �  �  `  F  ,    �  �  �  �  t  /  �  m  �   �  /  C  X  [  R  G  3      �  �  �  �  m  K  &    �  �  Q  �    *  ;  J  U  ^  a  Z  K  3  
  �  �  ^  .    �  �        �  �  �  �  �  �  t  M  "  �  �  i  �  r  �  y    �  4  <  M  X  ^  ^  T  D  2      �  �  �  |  U  "  �  �  g  �  �  �  �  �  v  ^  >    �  �  �  �  Y  .    �  �  v  F  �  �  o  U  9    �  �  �  z  O  $  	  �  �  �  �  �  @  �  $  *  0  6  .  #    	  �  �  �  �  �  �  r  Y  ?  "     �  ~  |  y  t  m  f  X  H  -    �  �  �  b  (  �  �  p  4   �  �  �      -  G  g  ~  �  v  c  I  %  �  �  }    {  �   �  �  �  �  �  �  �  �  �  �  �  �  �  �  {  q  g  \    �  k  �  �          �  �  �  �  o  @    �  W  �  o  �    F  B  S  c  S  B  5  ,      �  �  �  �  �  �  h  5  �  S  �  �  �  �  �  �  �  k  O  +  �  �  t  9  �  �    m  �  D  E    #  B  a  t  x  z  u  g  R  6    �  �  �  8  �    x  �  �  �  �  �  �  �  �  �  �  �  k  E    �  H  �  �  �  )   V    	       �  �  �  �  �  �  �  �  �  �  �  �  �  ~  n  ]  �  �  �    x  q  j  i  l  n  p  r  u  t  m  f  `  Y  R  K  ,  5  .      �  �  �  �  g  F  A  "  �  �  D  �  �  ;  �  {  �  �  �  �  �  �  �  �  �  x  a  =    �  �  U    �  �  &        �  �  �  �  �  �  �  �  �  �  �  �  �  �  R    �  �  x  f  N  4    �  �  �  �  p  L  (    �  �  �  l  =  	A  	A  	$  �  �    E  �  �  �  s  R  =  �  �  i        �  ,  �  �  �  w  c  O  9  "    �  �  �  �  �  p  I  "  �  �  �  �  �  �  �  �  �  v  _  A    �  �  ]    �    �    H  P  .  (      �  �  �  �  ~  P  !  �  �  �  V    �  ]  	  �      �  �  �  �  w  Q  (  �  �  �  k  9    �  �  �  �  �  4  8  2      �  �  �  �  �  �  �  r  a  L  4       �     �  �  �  �  �  �  �  t  g  Z  J  6  "          �  �  �  "  �  �  e  1    �  �  o  5  �  �  �  f  1  (  �  ~  !  �  �  �  �  \  !  
�  
�  
q  
H  
  	�  	h  �  F  �    �  �  /  �  <  c  }  �  �  �  �  �  �  o  Z  F  5    �  �    �  %  �  �  �  �  �  �  z  k  [  J  8    �  �  �  �  �  �  �  �  �  	X  	�  	�  	u  	S  	'  �  �  f    �    �  I  �  h  �  �  &  H  a  K  7  "    �  �  �  �  �  [  )  �  �  }  H    �  �  _  �    "  T  B    �  �  o  1  �  �  3  �  d    �  5  R  R  �  �           �  �  �  �  ]  *  �  �  �  {  K    �  w  �  �  �  �  u  d  R  @  -       �  �  �  �  �  x  _  G  .    �  �  �  �  �  �  �  ~  m  S  9       �  �  �  �  b  ?  p  �  �  �  �  �  �  �  �  t  :  �  �  g    �  "  �  �  �  Z  �  �  e  D    �  �  C  �  �  9  �  Z  �  r    �  (  �  [  X  T  I  <  *    �  �  �  �  �  i  I  (    �  �  �  Z  �  �  �  �  q  X  >  #    �  �  �  �  �  �        %  5  \  P  C  5  %      �  �  �    J    �  F  �  �    �  