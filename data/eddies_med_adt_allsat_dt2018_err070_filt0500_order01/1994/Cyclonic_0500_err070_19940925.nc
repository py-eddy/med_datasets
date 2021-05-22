CDF       
      obs    D   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?�             �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M��   max       P�hd       �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ��v�   max       <��
       �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>ٙ����   max       @F��G�{     
�   �   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��
=p��    max       @v~fffff     
�  +|   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @/         max       @Q`           �  6   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�&        max       @�@           6�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �\(�   max       <e`B       7�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�5�   max       B5?�       8�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A���   max       B5A       9�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >��   max       C��       :�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >[��   max       C��-       ;�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          ~       =   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          ?       >   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          )       ?$   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M��   max       P       @4   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?���Fs�   max       ?�9����D       AD   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       �+   max       <��
       BT   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>ٙ����   max       @F��G�{     
�  Cd   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min                  max       @v}�����     
�  N   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @%         max       @O�           �  X�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�&        max       @���           Y,   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         ?�   max         ?�       Z<   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�������   max       ?�,�zxl#     @  [L   4            
   !      +         5               "   *      -   c                                                   %   	      	      	   !   
   [               A   	      4         2   ?         
   9   $            }   %   O�I_N�ƮN�G NA��OTC|OlkvNc@PP@3|N��O!�(P�hdNc�O��M��O�;[PyaO��'N�6�O��P���NWŀN�mO�P~�O���N�7�OO��N��	O0TN.�-O�V�O�N�N�HN/}�N�NNj�O�,�O�|N��N���N��!N/k�P�OX��Pj�N�S:N�ƣO+�O�y�O�Nd?�NL��O�UN:��N��5P#4�P��OY�2N�2N���O���O�)N!�N�2O���Pd�xO$�N��
<��
<��
<���<�C�<u<T��<t�;ě�;�o:�o�o���
���
�ě��t��t��49X�D���D���T���e`B�e`B��C���t����㼣�
���
��9X��j��h���\)�t���P��P�,1�,1�49X�49X�8Q�<j�<j�<j�D���D���H�9�P�`�T���e`B�e`B�e`B�ixսixս}󶽅���7L��7L��\)��\)������������������w���T��Q콺^5��v�����������������������������������������)3-)& 	^amwzsmjaV^^^^^^^^^^�"),/30)��� 6BLO[b\ZZYOB6)))*6ABOYOB6)(#))))))����)5CFGB.�������	
#0480/#
				Z[^grt��������tgf[YZt����
! 
������ult\anrstnaYX\\\\\\\\\\xz��������������|zyx��������������������BNafikgd[NI;5.��������������������#/IYZapxnaUH</#st}���������������ys
&6CORXZYWOC6*
���/HGOM<:F<#�������KOR[ehtuthe[WONNKKKK��������������������"/;T[_`^TLH;/"������&���������
)?BINIGIKIB605BNQVNJB50000000000_gpt�����������~tg`_����������������������������������������kt������tpkkkkkkkkkk������������������������������������7<INLID<977777777777zz�����������zzzzzzz$)*669:66)(tz�������ztnttttttttsw~��������������}rs)5>BFBB5)���!!������MT]aceaaTPJKMMMMMMMM�������������������� ���        #0Ibnz~}xkb<
������

 ���������&6ALUn������nXH#IO[dhjjjhhh^[UOKIIII�����������������������������������������������	
��������������������������	)*/.)������������������������������������������tnkot�������������!)-5:BDFB53)Rb�������������tg\XR��������������������� 
#0<<6+'"
 ��<IMUbknppnjbUTII<<<<SUadkijgbaUQOOMRSSSS#+GHU\]aijaS</#<AHUnz�����znaUH<78<rz������zorrrrrrrrrr 

#,)%##
�      ���������������������4@N[fXSQ=#����BBGO[ahhomhf[OIB8:=B�����������¾µ´¿�������
��#�/�4�2�!�
����¦¦²§¦��v�s�l�h�s���������������������ìéâìù������üùìììììììììì�������)�5�B�N�W�X�\�[�N�G�B�)���S�F�5�5�:�;�;�F�S�_�l�x����~�v�p�l�_�S�#�"�#�(�/�1�7�<�@�B�H�>�<�/�#�#�#�#�#�#�Z�K�>�1�6�;�;�A�N�s���������������s�g�Z�����������������������ļʼʼ˼ʼ����������������������������������������������˾A�7�'�1�P�o�s�����ľ������žѾؾ;��f�A�5�,�+�5�A�N�V�W�N�A�5�5�5�5�5�5�5�5�5�5�s�l�g�f�h�l�n�s�����������������������s�����������������������������������������y�s�m�r�����������Ŀȿ����������������y�N�F�:�?�Z�s�����������������������s�Z�N�)���#�/�<�F�H�U�[�^�i�k�p�y�v�U�H�<�)��������������������$����������׾;˾׾���	��"�/�:�?�@�?�:�.���ýàÂÊìù��������)�>�C�4�)�������ý�Y�S�Y�]�e�k�r�r�t�{�~���~�s�r�e�Y�Y�Y�Y�����������!�#�*�!�!��������������������������������������������������,�	������	��/�H�O�V�h�r�q�m�e�Y�H�;�,���z�s�^�H�A�H�T�a�z���������������������a�U�T�_�a�m�r�z�����z�m�a�a�a�a�a�a�a�a�m�i�a�_�T�T�Q�P�T�`�a�m�z�����������z�m���������������ļʼռּ����ּּʼ����U�a�n�r�y�zÄ�z�x�n�f�a�U�H�<�3�;�<�H�U�����������	���	���������������������ݿĿ��������������Ŀӿ�����&�������b�X�U�b�vŔŭŹ����������ſŭŠŔŇ�n�b�����(�/�4�;�4�(��������������������������������������������ìáàßàììù��������������ùìììì����������������
��
������������������¦�|�m�h�r¦²������������¿¦���������������������������
��������������� ����!�(�.�0�/�.�$�!������ѿĿѿ׿ݿ����������ݿѿѿѿѿѿѿѿѿ.�&�"�����"�.�/�;�G�J�T�X�T�G�;�.�.�ʾľʾѾ׾޾������׾ʾʾʾʾʾʾʾ��������{�q�f�^�^�g�s�����������������������������)�,�5�=�F�K�N�B�5�)���������m�T�G�>�P�z���������������������������������������������������������������(�5�5�5�5�(�%�������������	�������������������� ���"�'�(�"��	�$�����$�&�0�=�I�V�b�l�s�q�b�I�=�1�$�һ»������л����'�4�@�C�@�9������Ҽ@�>�4�1�/�2�4�@�M�Q�N�M�F�A�@�@�@�@�@�@�û����ûлܻ�����ܻۻлûûûûûüA�B�M�e��������ʼּ����ּʼ����r�AƎƎƖƎƁ�u�n�o�uƁƎƎƎƎƎƎƎƎƎƎ�[�T�[�a�h�p�t�~āččĎčĉā�~�t�h�[�[Ěčć�yĊčĚĦĳĿ����������������ĿĚ���������������
�#�<�U�o�|�w�b�W�<��
���Ľ��������������������нݽ����ݽнĽ���������������������������������������E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�FFF$F1F4F=FFF>F.F$FFE��������z�u�q�z�������ɾооɾȾɾ�������F$FFFFF$F1F4F7F1F$F$F$F$F$F$F$F$F$F$D�D�D�D�D�D�D�D�EEEEED�D�D�D�D�D�D캤�������������ɺֺ�����������ɺ����z�j�^�U�H�0�.�<ÇÓì����������à�z�����������������ùϹڹܹ����ܹϹù������������������ĽƽнԽнĽĽ��������� 4 D U < 9 & r 9 S 4 K 6 ; < d ? > >  5 c J . 5 O - X d ( ^ j = J j ; @ E \ Q T v ` E k q a 0 �  0 h Y P 2 n H 4 ; G c " . / 2 : q ) d    �  �  R  �  �  �  L  �  b  �  f  R    m  t  �    �  �  �  '     �  f  �  �  	  q  �  �  �  '  �  �  h    C  9  �    k  �    �  �  �  ,  ,  !  �  z  _  F      �  �  �    O  �  6  �    �  U  ��o<T��<e`B<t�;��
���
;o��P�o��`B�]/�#�
�D���o��/�0 ŽT����C��e`B��S��������
�'H�9�\)���ͽo���H�9�+�T������#�
��w�]/�Y����w�Y��y�#�Y��Y��aG������ixվV�ixս����t���-������u��/��O߽��������vɽ�{��+��S����T������;d�\(��J���B��BׄB��A�@�B�KB�B�gBZB%'B	��B!�B}BW�B5?�B��B"#B$rB
��B0B�pB`�B+|�A�5�B˭B_�B�0B
��B!,�B!M�B
-B*�EB�mB&w�B ��BB 
&B�7B��B.V�A��KB8�B?�B&��B?-BՠB5,B�B!<�Bc)B��B�gB)��B+8,B	��B�%B
�*B8(B$�3B'{�B�3B��B��Bo�B5dBU[BR�B�{B�B��B>�B�[A��BF�BǈB��BInB$ȾB	BB�vB@�B��B5AB�B?�BB
��B0	.B�gB@PB+��A���B1>B@/B��B5tB!��B!=�B
�B*��B��B&P{B |TB �A��#BQ*B;�B.o�A���B=B@B&��B۸B�gB;�B�B!AhB�B�:B�aB)��B+C�B	��B�PB
I�B?^B$��B'��B6iB@gBCmBKfB=BBB0B@:B:dB?�A���A�t�AE��A�HJA�@8@�[�Aª�A��.@�l�A�}�ADW�A���A��lAJ��Ar(]A���A�_TA�POA[�AИ9?�z�@X��A���A�i>A�U�A�[�A�4c@���A���AY%yA}�
A�>�A6�DBumA͘�A�gpA�p\A��A
��A}��Aa��AT!�A�ԉA�HXA�Z�A�-�A�4�A�e�BA@�Y|@ш�@��@�;�B1A�^NA��A��A&�HA!��C�0�C���AJn�C��C�@�@7i�A��>��A#��A���A��!AE�A�ruA�r:@���A�xA��@� _A�|WACA��A�IAJ�6Ap�@A�~UA�}�A��AZ�qA��?�V�@_8�A�E�A��A��vA��A�lTA ��Aŀ�AY��A{�A�~bA7�1BH�A�~�A��A���A���A �A~֐Aa"AS-A�z�A�2{A��lA���A�b�A��B�T@�C�@�3�@��@���B@�A܋gA���A��YA&hA �4C�7�C���AH��C��-C�8�@<�[Aʇ�>[��A%6   5         	      "      ,      	   5               #   *      -   c                         	                           %   
      	      
   "   
   \   	             A   
      4         3   ?         
   :   %            ~   %      #                     +         ?               '            9            )                     )                  #                  )      =               #         '         )   )               !            9                                       )                           #                                 )                  #                  )                              '         !                  !            !      OjN�ƮN�G NA��OTC|NӜNc@PO��CN��OPNc�O��M��Ob�:O�A�N�-N���Oa��O�B�NWŀN�mO�O���O���N�7�O<
.Nm �O'��N.�-O�V�O�N�N�HN/}�N�NNj�O�,�O�|Nˎ:N���N��!N/k�P�OX��O��N�S:N�ƣOO��-O�#NE_�NL��O�)�N:��N��5O�6O�*�OY�2N��N���O[��O�)N!�N�2O���O�?�N��'N��
  s  1  G    ,  �  5  t  �  A  �    !  /  �  �  Z  �  �  	j  k  �    �  �  G  �  t  [  �  "  ^  �    Q  �  �  ]  �  S  /  �  ?  x  �  �  R    �  �  y  Z  �  �  �  5  �  	  �  �  	�  �  �  f  �  -  f  u��o<��
<���<�C�<u��o<t��49X;�o%`  ��j���
���
�ě��49X��C��\)�T�������T���e`B�e`B��C���`B���㼣�
��1�ě��ě���h���\)�t���P��P�,1�,1�49X�<j�8Q�<j�<j�<j�D����1�H�9�P�`�aG��ixս�hs�ixսixսu�}󶽅����w��9X��\)��t������1�����������w���T�+��񪽾v�����������������������������������������)3-)& 	^amwzsmjaV^^^^^^^^^^�"),/30)���()16BJOQRQODB@6+)$(())*6ABOYOB6)(#))))))����%021,)�����	
#0480/#
				Z[\agnt��������tga[Z����������������|{|�\anrstnaYX\\\\\\\\\\xz��������������|zyx��������������������)BNW_ghg][NK>51"��������������������#)/6<>@<</#��������������{t����)6CFLOQOIC6*���
#/52&! 
������KOR[ehtuthe[WONNKKKK��������������������"/;T[_`^TLH;/"����������������
)?BINIGIKIB605BNQVNJB50000000000cgs�������������tgac����������������������������������������kt������tpkkkkkkkkkk������������������������������������7<INLID<977777777777zz�����������zzzzzzz$)*669:66)(tz�������ztnttttttttsw~��������������}rs)5>BFBB5)��������MT]aceaaTPJKMMMMMMMM�������������������� ���        #0Ibnz~}xkb<
������

 ���������Uanz�������zaUPKKLNUIO[dhjjjhhh^[UOKIIII����������������������������������������������
�����������������������������)*.-)������������������������������������������tnkot�������������!)-5:BDFB53)Z\g�����������tgb]ZZ��������������������� 
#0<<6+'"
 ��EIPUbjnoongbYULIEEEESUadkijgbaUQOOMRSSSS#/<HUWTUZVHE</# <AHUnz�����znaUH<78<rz������zorrrrrrrrrr 

#,)%##
�      �������������������� )8@B@:0)�� ABCMO[_hjih_[OCBAAAA�������������������������
������
������¦¦²§¦��v�s�l�h�s���������������������ìéâìù������üùìììììììììì�������)�5�B�N�W�X�\�[�N�G�B�)���S�P�F�C�D�F�K�S�_�l�p�x�y�x�t�l�j�_�S�S�#�"�#�(�/�1�7�<�@�B�H�>�<�/�#�#�#�#�#�#�g�N�J�D�C�A�N�Z�g�s�����������������s�g�����������������������ļʼʼ˼ʼ����������������������������������������������׾Y�L�F�C�G�M�Z�s�����������������s�f�Y�5�,�+�5�A�N�V�W�N�A�5�5�5�5�5�5�5�5�5�5�s�l�g�f�h�l�n�s�����������������������s�������������������������������������������y�s�p�v�����������Ŀſ����������������Z�N�F�I�Z�g�s�����������������������g�Z�<�1�/�,�/�7�<�H�U�U�a�]�U�H�<�<�<�<�<�<����������������	������������ھܾ����	��"�*�2�4�1�.�"��	��ìèãààãêù������'�&� �������ì�Y�S�Y�]�e�k�r�r�t�{�~���~�s�r�e�Y�Y�Y�Y�����������!�#�*�!�!��������������������������������������������������/�"������"�/�;�H�T�[�e�_�Z�T�H�;�/���z�s�^�H�A�H�T�a�z���������������������a�U�T�_�a�m�r�z�����z�m�a�a�a�a�a�a�a�a�m�k�a�V�T�R�Q�T�a�m�z���������������z�m���������Ƽʼּ�ڼּҼʼ����������������U�H�<�<�4�<�H�U�a�n�q�y�zÂ�z�v�n�e�a�U�����������	���	���������������������ݿĿ��������������Ŀӿ�����&�������b�X�U�b�vŔŭŹ����������ſŭŠŔŇ�n�b�����(�/�4�;�4�(��������������������������������������������ìáàßàììù��������������ùìììì����������������
��
������������������¦�|�m�h�r¦²������������¿¦���������������������������
�������������������!�%�.�/�.�-�"�!�������ѿĿѿ׿ݿ����������ݿѿѿѿѿѿѿѿѿ.�&�"�����"�.�/�;�G�J�T�X�T�G�;�.�.�ʾľʾѾ׾޾������׾ʾʾʾʾʾʾʾ��������{�q�f�^�^�g�s�����������������������������)�,�5�=�F�K�N�B�5�)���_�Q�O�V�d�m�z���������������������z�m�_��������������������������������������������(�5�5�5�5�(�%���������������	������������������
���"�%�%�"��#����$�'�0�=�I�V�b�k�r�p�b�U�I�=�0�#��޻̻ûŻɻлٻ������$�(�������4�3�0�3�4�@�M�P�M�M�D�@�4�4�4�4�4�4�4�4�û����ûлܻ�����ܻۻлûûûûûüY�M�M�h��������ʼ���޼ּʼ������r�YƎƎƖƎƁ�u�n�o�uƁƎƎƎƎƎƎƎƎƎƎ�[�T�[�a�h�p�t�~āččĎčĉā�~�t�h�[�[ĦĚĒČĄ�āčĚĳĿ������������ĿĳĦ�#��
�������
��#�<�D�U�Z�[�V�L�<�0�#�Ľ��������������������нݽ����ݽнĽ���������������������������������������E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�FFF$F1F<F=FAF:F*F$FFE��������z�u�q�z�������ɾооɾȾɾ�������F$FFFFF$F1F4F7F1F$F$F$F$F$F$F$F$F$F$D�D�D�D�D�D�D�D�EEEEED�D�D�D�D�D�D캤�������������ɺֺ�����������ɺ���Ç�z�q�k�i�n�zÇìù��������������àÓÇ�Ϲǹù����������ùϹѹܹ��ܹڹϹϹϹϽ����������������ĽƽнԽнĽĽ��������� , D U < 9  r ' S . I 6 ; < ` B  D # : c J . : O - Y b ) ^ j = J j ; @ E \ ? T v ` E k & a 0 ~   Z Y H 2 n 6  ; , c  . / 2 : ? " d  �  �  �  R  �  �  �  x  �  .  �  f  R      o  �  �  �  N  �  '     �  f  �  �  �  d  �  �  �  '  �  �  h    C  �  �    k  �    n  �  �  �    -  t  z  
  F    �  <  �  �    �  �  6  �    �  �  �  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  �    x  �  
  B  d  r  r  f  M    �  P  �    o  �  �  O  1  -  )  $          �  �  �  �  �  �  �  �    )  A  Y  G  ;  .  "      �  �  �  �  �  �  �  �  o  \  @  #     �      �  �  �  �  �  �  �  �  �  r  _  K  7  #      "  :  ,  #        �  �  �  �  �  �  �  �  x  Y  4  E  c  [  O  
    7  P  h  ~  �  �  �  �  �  g  7  �  �  r    �  A  N  5  -  %          �  �      1      �  �  �  �  �  h  �    7  D  N  V  h  r  q  h  S  2    �  �  r  $  �  _  �  �  �  �  �  �  �  �  �  �  �  p  ]  K  =  7  2  .  5  ;  B  ;  >  @  <  4  +         �  �  �  �  �  �  �  �    h  R    `  �  �  �  �  �  �  �  �  �  �  n  *  �  @  �  v    ?            �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  !        �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  /  ,  )  &  #                  �   �   �   �   �   �   �   �  �  �  �  �  �  �  �  �  �  �  {  X  .  �  �  �  L  �  �    �  �  �  �  �  �  �  �  �  �  �  �  �  s    �  �  �  9   }        �  �  $  "    #  ;  R  Z  I  (  �  �  k    W  i  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  z  l  �  �  �  �  �  �  �  �  �  �  �  �  �  �  H  �  �  ;  �  �    l  �  L  �  �  	+  	H  	g  	V  	9  	  �  �    i  �  \  W  �  k  H  I  z  z  n  `  G  *    �  �  �  q  K    �  �  O    �  �  �  �  �  �  �  �  �  �  �  �  �  r  V  :       �   �            �  �  �  �  �  �  Y  0  	  �  �  {  B    u  :  c  �  �  �  �  �  �  �  �  q  ?  �  �  B  	  �  U  �   �  �  �  �  ~  a  D  )    �  �  �  �  q  E    �  �  �  Q   �  G  F  D  C  A  =  9  5  1  +  &  !        �  �  �  �  �  �  �  �  �  s  _  E  ,    �  �  �  �  w  R  1    �  �    n  j  f  i  o  r  h  ]  P  C  2    	  �  �  �  2  �  �  m  T  Y  O  ?  ,    �  �  �  x  I    �  �  ;  �  M  �    
  �  �  �  �  �  �  �  �  �  �  �  �  �  �    u  k  `  V  K  "    	  �  �  �  �  �  e  =    �  �  �  �  W  "  �  �   �  ^  Y  T  N  I  5    �  �  �  t  O  ,    �  ~    �  b    �  �  �  �  �  �  �  �  �  �  �  {  p  f  [  Y  Z  Z  [  \      �  �  �  �  �  �  �  �  �  �  �  �  �  �  z  n  c  X  Q  L  K  I  B  9  -        �  �  �  �  �  �  }  U  '  �  �  �  �  �  �  �  �  �  �  �  �  {  e  O  8  !  	  �  �  �  �  �  �  i  F  6  9    �  �  �  �  �  <  �  �  )  �  ~  P  ]  W  Q  P  O  N  M  I  B  7  )      �  �  �  �  �    �  �  �  �  �  �  �  �  k  F     �  �  �  z  T  '  �  �  =  �  S  F  9  )      �  �  �  �  o  F    �  �  �  {  Q  #   �  /      �  �  �  �  s  I  5  $    �  �  �  �  �  a  <    �  �  �  �  �  r  Q  .  	  �  �  �  o  E    �  �  �  ]  -  ?  %    �  �  �  �  �  x  U  .    �  �  �  �  n  :    �  x  r  l  a  V  Y  _  V  C  -    �  �  �  �  j  <    �  �  �  �  �  �  e  �  �  �  �  �  �  n    �  W  �  9  T  �  m  �  �  �  �  �  �  �  �  �  u  i  \  M  4    �  s    �  C  R  G  4      �  �  �  �  W  $  �  �  f    �  �  C  �    �  �  �     �  �  �  b  0  �  �  �  �  k  /  �  a  �  �  =  �  �  �  h  =    �  �  �  �  L    �  �  8  �  ;  �  �  �  0  K  d  �  �  �  �  �  `  )  �  �  A  �  q  �  M  �  �  �  x  y  y  p  e  W  H  5      �  �  �  �  _  4    �  �  e  Z  W  T  Q  M  J  G  A  9  1  *  "         �   �   �   �   �  �  �  �  �  �  �  X  )  �  �  �  Q    �  `  �  �  *  �  �  �  �  �  �  �  �  l  R  9  (      �  �  �  �  �  j  G  %  �  �  �  �  |  W  4    �  �  �  m  *  �  �    �  �  @  �  �  p  �    5  +    �  �  �  e  *  �  �  3  �    >  x  h  �    K  t  �  �  �  �  �  �  �  ]     �  H  �  .  a    z  	  �  �  �  �  �  ~  W  *  �  �  �  F    �  u  !  �  9  �  �  �  �  �  �  q  S  3    �  �  �  x  H    �  �  *  �  f  �  �  �  �  ~  h  P    �  �  �  �  �  �  w  W  ;    �  �  	  	|  	�  	�  	�  	�  	{  	p  	o  	W  	  �  \  �  2  �  �  ^  �  P  �  �  u  Z  8    �  �  �  O    �  �  L    �  O  �  l  �  �  �  �  ~  w  q  k  d  ]  V  N  G  @  6  (            f  `  Z  S  I  :  $    �  �  �  3  �  �  +  �  �  b  �    �  �  p  U  ;  $    �  �  �  �  �  S    �  �  &  �  N  �  �  ^  �  �       %    �  �  �  K  �  b  �  
�  	�  Z  �  �  �      0  E  W  c  e  \  >    �  �  ~  M    �  �  3  �  u  b  L  5    �  �  �  �  �  y  _  +  �  q  +  �  �  J   �