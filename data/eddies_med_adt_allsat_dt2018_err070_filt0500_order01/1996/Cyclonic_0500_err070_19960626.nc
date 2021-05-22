CDF       
      obs    P   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?����+     @  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M�)�   max       P��     @  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ��
=   max       <�1     @   ,   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?W
=p��   max       @F��Q�     �  !l   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       �׮z�H    max       @v��Q�     �  -�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @+         max       @P�           �  :l   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�d        max       @�:`         @  ;   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ���   max       <T��     @  <L   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A���   max       B4��     @  =�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�x�   max       B4�Y     @  >�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >PD�   max       C��"     @  @   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >>j|   max       C���     @  AL   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          ^     @  B�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          A     @  C�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          7     @  E   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M�)�   max       PqA     @  FL   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?������   max       ?�m��8�Z     @  G�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��
=   max       <�1     @  H�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?\(�\   max       @F��G�{     �  J   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       �ҏ\(��    max       @v��Q�     �  V�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @$         max       @P�           �  c   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�d        max       @���         @  c�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         BR   max         BR     @  d�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��PH�   max       ?�m��8�Z        f,               .                                                    ]                                  .   ?                           
                            0            !   2   
   
         1      
            	         #                           	N�NY�N�=N'5�O�ghN�2MN��O���O]^�N�w�N]� N`��N��O���O���O�=�NH��Oʞ>NZαNW#�N{e�P��N��;O��YN��M��NW�DN/�pO�M�)�N�e�N�t�Oڷ�O�P"NJ��NN�'�Ok8�O�F�P3�O'd�O��FO��vOkˌO2OP(�SN�O�ƥNi�5N�7O]'�O�G�Nc#�N?��O�� O�ʝO���N5�NY�O 5O��P-CO���N7̑Og�N�<
N2��N�	sN�WN
��O��>O��_Oc�ROCj�N8�'M�DN2�=O(]�N�w�N��<�1<D��<D��;o:�o�D����o���
�t��t��49X�49X�D���e`B�e`B�e`B�e`B�u�u��o��C���C����㼛�㼛�㼛�㼛�㼛�㼣�
���
��1��1��1��1��1��1��1��j��j�ě��ě����ͼ�����/��/��/��/��h��h��h���o�C��C��C��\)�t��t��t���w��w�0 Ž49X�49X�49X�8Q�8Q�<j�@��L�ͽT���u�u�u�}󶽡����vɽ\����
=�������������������� #
MO[ehniha[VOMMMMMMMM
"#/#

�
#.<FQTI</#	����Xanz����znaXXXXXXXXX��������������������
#0<IUaca\UI<# EHTadmoonooomTNF@>@Exz�����������zyxxxxxdhjtw�����toihdddddd������������������������������������������#/10230(#
�������������������������������� ������������������������������{���������������zus{��������������������	$$ 										����������������������������(,%����������������������������5@>:52$�����)5852)������

�����������#/;<DC<1/#"����������������������������������������z{����{{ywzzzzzzzzzznfbUJIFEGIMUbnnqqnnnGHMTX[]\ZVTNHFDBBBGGTamz��������zmaYQMMT���������������������������������������������������������	)69;86)"				W[gt�����������sg[QW 	#/9==;8/#
��� _m}���������zvmb]ZZ_JNR[_got���{tg[NJGJz���������������zuuz}�����������������v}������
!
���������
#/7DF<.#
 ��#0Iv������{gI<4!#����������������������������������������NOZ[hlt|tph[RONNNNNN���������������������������������������������������������|����������z||||||||��������������������+2<Ibr{~}wrrebU<3/'+)6B[hsqkheba[TOKA74)5<Hanz~z|�vnaUH<725IO[[cc[QOKIIIIIIIIIIXbnx{{{vndbZXXXXXXXXR[fgt��������tmg[PRR��������������������RT[gt�������ztg[NLLR�������		���������)46766)";MU[ht����wmkli[OC;;����������������������������������������dgst�����������tgfdd��������������������$#!���� #-/4)����gjt�������������|tjg�����������������������������������������������������������������������������������������������anuz~zypncaUHA?BHRUa
#(.##"
	FHUU^WUHG@FFFFFFFFFF�
���
����#�/�4�8�8�/�'�#��
�
�
�
��׺ֺҺֺ����������������Ŀ����Ŀǿѿݿ�ݿܿѿǿĿĿĿĿĿĿĿļ������������������Ƽ�������������������ŝŕŜŦŭŹ���������� ��������Źŭŝ�H�G�C�B�B�H�U�Z�a�`�[�U�H�H�H�H�H�H�H�Hììèìù��������ùìììììììììì���������z�s�u�u�w�������������ǽȽ�����ā�|�p�p�tĈčĚĦĳĿ������ĿĳĦĚčā����������������������������������������H�>�<�8�<�F�H�U�a�a�c�a�U�Q�H�H�H�H�H�H�L�H�B�E�L�T�Y�e�o�e�c�Y�L�L�L�L�L�L�L�L�ѿ˿Ŀÿ����������ĿϿѿݿ߿ݿۿٿܿѿ��g�]�B�4�*�1�5�B�N�[�t�t�g�������������Ľн������"������н��9�4�(�+�(�&�4�A�M�f�l�s��������f�Z�M�9�T�R�G�E�D�G�J�T�Z�[�`�b�`�T�T�T�T�T�T�T�������������Ŀѿ��������
����꿸������ĿľĿ�����������������������������<�9�7�<�H�U�a�Y�U�H�<�<�<�<�<�<�<�<�<�<�������������������������������뼽�����Y�I�O�y�����ʼ���!�/�)����ʼ��Z�R�N�A�:�A�N�R�Z�g�o�l�g�]�Z�Z�Z�Z�Z�ZùàÇ�z�t�n�x�zÇÓØàù������������ù�s�i�g�g�f�g�o�r�s�z�����������������s�s�s�o�f�Z�W�Z�f�s�w�����s�s�s�s�s�s�s�s�ݽսԽݽݽ�����������ݽݽݽݽݽݽݽݼ����������������������������������������^�b�h�{ńŕŠŭŹž������ŹŠŔŇ�{�n�^����������������������M�Z�`�e�Z�W�M�A�4�/�(��!�(�(�4�A�M�M�M�N�M�N�O�Z�g�s�����������������s�g�Z�N�NưƘƑƔƜƧƳ����������������������ư�������������������$�0�7�=�6�0�$����F$FFF$F.F1F=F>FJFNFJF=F1F'F$F$F$F$F$F$F=F9F=FAFBFJFMFVFaFVFKFJF=F=F=F=F=F=F=F=��ܹܹѹֹڹܹ�����������������������y�r�n�l�o�z�����������������������������s�l�f�i�s�����������������������������������	�$�/�H�T�^�[�`�[�T�N�/�"�	�����������������������*�7�B�;�6�)���������������������)�6�A�7�1�!������������������������	���)�4�0�.�)�������������z�z����������������������������������$�&�)�6�B�^�O�H�=�6�)�"��e�]�U�A�5�Z�s�����������������������s�e����������#�����������������������������ʾ����	�����׾ʾ����ù¹����������ù̹ϹӹչϹƹùùùùùùϹʹιϹ۹ܹ����߹ܹϹϹϹϹϹϹϹϾ��������������������ʾҾ׾ܾ۾ؾ׾;������x�r�i�l�x���������ûܻ����û��������A�>�6�5�4�5�A�N�T�X�Y�N�A�A�A�A�A�A�A�A�g�c�d�g�n�s�������������s�i�g�g�g�g�g�g�û��������������ûл���������л����}��z�{�����������������������������������������������ùϹ����������ܹù��Y�N�Y�Y�e�r�x�r�n�e�Y�Y�Y�Y�Y�Y�Y�Y�Y�Y�-�&�%�-�8�:�=�F�N�F�E�:�-�-�-�-�-�-�-�-���������������	��������	�����h�d�_�f�h�tāčĚĦııĦěĚčā�t�h�h²�z�w�x²������
���������²�����������������ʼּ����޼ּʼ������лͻɻлӻܻ����߻ܻлллллллл��������~�~�x�v�x���������ûл߻ܻлû�ŹŸűŹŻ��������������������������ŹŹ�l�`�`�`�k�l�l�v�x�y�x�w�l�l�l�l�l�l�l�l�<�8�/�/�/�/�7�<�H�U�a�f�a�U�Q�K�I�H�<�<�*�$�*�4�6�C�G�O�U�O�C�6�*�*�*�*�*�*�*�*�Z�T�M�H�M�Z�f�s�u�s�f�^�Z�Z�Z�Z�Z�Z�Z�Z�:�.�!����������!�:�������������l�S�:��ĿĴīĮĳĿ�����������
��"��������H�=�7�1�1�=�I�V�b�o�ǈǉǉǈ�{�o�b�V�H���
���������������
��#�+�-�1�0�)�#��T�J�K�T�`�m�n�m�m�`�T�T�T�T�T�T�T�T�T�TììùþùìììàÙàììììììììì�������������������ľɾ������������������������������Ŀѿӿݿ޿ݿؿѿĿ���������EEEE!E*E7E@ECEPE\E]E\EXEPEEECE7E*EEECEBECECEPE\EcE^E\EPECECECECECECECECECEC < < b X U m ' A N < 8 > k M ` ! i  ^ 6 ^ O O Y Z z T � R E " o % % W v 4 [ F P X D 7 V � 4 H 8 K w ( # 9 k B P + U > O K O Q 3 | X c Z S R x D . = 9 ? p % r 6    E  V  j  �     /  �  �    n  q  �  c  {  +  �  �  n  m  �  �  �  �    "  �  �  U      H  �    }  p  �    "  �  �  �  V    �      �  �  �  �  1  w  �    ^  �  <    <  X  B  G  S  O  	  �  �  S  5  R  z  �  �  Y    f  k  	  1<T��;��
<#�
�o�8Q�e`B�#�
�\)��`B�u��o�u�u��P�8Q��P��C��<j���㼬1���
��G��ě��L�ͼ�����1�o��1�\)��9X�+��`B��+��1��`B���ͽ�w��w�P�`�<j�T���aG��t����#�
�H�9���@��'C��}󶽡����P��㽁%��C��� Ž8Q�<j�L�ͽ�o��j�����Y���+�e`B�T���]/�H�9�Y��� Ž�1��-��t���+��j�Ƨ�����#��l�B GB_aBB�B$ŅB�B �B��B&A��)B@BE�B"�9B,B&B?B!|B"�eB,�B!<B�B�PB�\B,��B�B��ByB��BJ�B l1B&DB)	�B'xeA���A��}B"�BAB��B�$B
]#B�B ��B	yB �\B�GB��B��B'`hB)�Bb2B��B��B4��BƶB
��BøB'B�B�BȎBPB(OB	��B�\B
	RBn�B=:B~EB�B ��B
xB
ʴB�B��B
boB��B��B�2B?�BF�B�BQfB�B��BRNB5�B$�WB��BTB��B%��A��sB5�BX�B"G2B,݆BACB!BUB"��B,O�BBB��B�fB�BB- 'B��B��B��B�AB��B }�BziB(��B'}oA�x�A�}�B:�B?ZBB�B�B
��B�QB EB	:HB ��B� B�TB<tB'LB)�B��B�B@�B4�YB�TB
��B?�B'?�B:�B?9B<�B(9_B	��B�B	�B=�B@B*�B>uB �B
AB
�BB�B:�B
@�B��B�PB�=B@B%�B;�B?iB]�A�&�@D}Az�L@�	LA�CFA�Ä́LA �oA���A��A��?��Ay�A���A)�^A=׶Af��A}1IA�J`AČ5A��6@�V�A�, A��yA��TAB0>A,[B@���A�ƌA2�[A;
rA�t�B6�B�C�ۘC��"?"�A�0�A���A�aA�o�A�`A�k�A���A���A�R^A�eJAQ��>Wc�>���AM��@���A���A��U@�qA�̫>PD�?� �@y:qAZ7VA�s6A�W@��e@���@�ҋA���@�\>Aĝ�B �6A?��A��A出ByA��+AhYA�v�AMfTAw��C���C��kA�a�@DSAy��@��A���AŌ�A�u�A!.A�e�AЁ�A��?נ�Ay�gA���A*�NA>�Af��A}�;A�,A�n�A��A��A��"AˋhA��+AA
A-@��A�y�A2�*A;hA�i]BH�B��C��C���?_�A���A��oA�m�A�v�A�D.A��A�w�A��A��A�~�AR�>>j|>���AM��@�:A���A��@�:	A�yq>�[�?�=�@tH�A[�A܀�A�s�@�1e@�&�@�J�A��+@���AÖzB ��A@KA�rA��B�UA�LAh��A�}�AM��Awl�C��#C���   	            /                                                     ^                                  .   @                                                       0            "   3   
            2       
            	         #                           	               !                                                   A      !                           !   !                  )      !            )      #            '         )      #               -         !                  -                                          !                                                   7                                                      '                  )      #            !                              )                           -                           N�NY�N�=N'5�O�ghN�-N��O��O:��N�w�N]� N`��N��OvOqO�GN��NH��O��!NZαNW#�N
A�PqAN��;OK��N��M��NW�DN/�pOw�RM�)�N���N�t�O�+One"NJ��NN��aOk8�Og	PhO��O|�lO��JOkˌN�[�P(�SN�O�ƥNi�5N�7O6�O�/SNc#�N?��O``eOpO]�N5�NY�O 5N��P
�:O���N7̑OR%�N�<
N2��N�	sN�WN
��O��O��OIu8N��N8�'M�DN2�=O(]�N�w�N��  �    �  �  b  i    �  0    y  6  �  �  H  g  �  �  �  1  }  R  '  �  �  �    �  �  �  0  	  /  �  �  �  *    P  f  �  �  �  �  �  �  
  M  �  �  �  
  �    V  �  �  �  �  W  �    w  y  =  �  �  !  ~  �  L  f  �  �  d  I      �  r<�1<D��<D��;o:�o�o��o�t��49X�t��49X�49X�D����o���ͼě��e`B��t��u��o��t���������������㼛�㼛�㼛�㼬1���
��9X��1���0 ż�1��1�ě���j�����ͼ�`B����/��/����/��/��h��h��h�#�
�t��C��C��49X�8Q�D���t��t���w�8Q�L�ͽ49X�49X�8Q�8Q�8Q�<j�@��L�ͽ]/��%�}󶽃o�}󶽡����vɽ\����
=�������������������� #
MO[ehniha[VOMMMMMMMM
"#/#

�
#.<FQTI</#	����inz����znbiiiiiiiiii��������������������#0GU]`^ZUI<0#FHTahnnmlnmlaTQHB@BFxz�����������zyxxxxxdhjtw�����toihdddddd�������������������������������������������#//11.&#
 ���������������������������������������������������������������y|��������������~xuy��������������������	$$ 										����������������������������%'����������������������������
)11,)%�����)5852)������

�����������#/;<DC<1/#"����������������������������������������z{����{{ywzzzzzzzzzzHIOUbgnppnmebULIGEHHGHMTX[]\ZVTNHFDBBBGGSTWamz�������zmaUQQS���������������������������������������������������������)36965)W[gt�����������sg[QW	
#/27851/#
		[amz�����������mc^[[LNU[cgt~~ytrge[NMILLz��������������}zwwz������������������z�������
!
�������
#/2<@B><5/#!



#0Iv������{gI<4!#����������������������������������������NOZ[hlt|tph[RONNNNNN����������������������������������������������� ����������|����������z||||||||��������������������:=EIUbdmnnlhaVSIC<9:>BEO[hikjhd_[OMGCB?>9<HUamnnljieaUH<:769IO[[cc[QOKIIIIIIIIIIXbnx{{{vndbZXXXXXXXXR[fgt��������tmg[PRR��������������������NTVY_gt�������tg[RNN�������		���������)46766)"<NV[ht���~vlkli[OD<<����������������������������������������dgst�����������tgfdd��������������������$#!��").-0+����mt������������{tljm�����������������������������������������������������������������������������������������������anuz~zypncaUHA?BHRUa
#(.##"
	FHUU^WUHG@FFFFFFFFFF�
���
����#�/�4�8�8�/�'�#��
�
�
�
��׺ֺҺֺ����������������Ŀ����Ŀǿѿݿ�ݿܿѿǿĿĿĿĿĿĿĿļ������������������Ƽ�������������������ŝŕŜŦŭŹ���������� ��������Źŭŝ�H�G�F�F�H�U�U�\�[�U�H�H�H�H�H�H�H�H�H�Hììèìù��������ùìììììììììì�����~�v�y�{�������������ýĽĽ���������ā��s�sāčĚĢĦĳķĿ����ĿĳĦĚčā����������������������������������������H�>�<�8�<�F�H�U�a�a�c�a�U�Q�H�H�H�H�H�H�L�H�B�E�L�T�Y�e�o�e�c�Y�L�L�L�L�L�L�L�L�ѿ˿Ŀÿ����������ĿϿѿݿ߿ݿۿٿܿѿ��t�i�`�[�B�7�-�4�B�N�[�t�~�t�ݽսнĽ������������Ľнݽ��������ݾA�;�4�4�3�4�A�M�Z�f�h�p�p�f�Z�M�A�A�A�A�T�R�G�E�D�G�J�T�Z�[�`�b�`�T�T�T�T�T�T�T�ѿĿ����������Ŀѿݿ����	�
�������������ĿľĿ�����������������������������<�9�7�<�H�U�a�Y�U�H�<�<�<�<�<�<�<�<�<�<������������
���������������������뼽�����f�`�c�r���ʼ���!�)�*�%�� ��ʼ��Z�R�N�A�:�A�N�R�Z�g�o�l�g�]�Z�Z�Z�Z�Z�ZìàÍÇÀ�z�~ÇÓàåìùü��������ùì�s�i�g�g�f�g�o�r�s�z�����������������s�s�s�o�f�Z�W�Z�f�s�w�����s�s�s�s�s�s�s�s�ݽսԽݽݽ�����������ݽݽݽݽݽݽݽݼ����������������������������������������n�b�i�n�{ņŗŠŭŹŻſ��ŽŹŠŔŇ�{�n����������������������4�2�(� �%�(�-�4�A�I�M�Z�^�c�Z�S�M�A�4�4�N�M�N�O�Z�g�s�����������������s�g�Z�N�N��ƺƳơƙƞƧƳ������������������������������������������$�&�/�0�3�0�*�$��F$FFF$F.F1F=F>FJFNFJF=F1F'F$F$F$F$F$F$F=F9F=FAFBFJFMFVFaFVFKFJF=F=F=F=F=F=F=F=���ܹӹعܹ����������������������y�r�n�l�o�z�������������������������������t�s�n�p�s�������������������������������������	�%�/�H�T�\�\�X�^�L�/�"�	������
�� ������'�)�4�6�>�7�6�)�����������������������)�.�(�������������������������#�)�1�.�,�)��������������z�z���������������������������)��&�(�)�2�6�B�O�V�O�K�D�B�9�6�)�)�)�)�e�]�U�A�5�Z�s�����������������������s�e����������#�����������������������������ʾ����	�����׾ʾ����ù¹����������ù̹ϹӹչϹƹùùùùùùϹʹιϹ۹ܹ����߹ܹϹϹϹϹϹϹϹϾ����������������������¾ʾҾӾϾʾ¾������x�m�x�����������ûлܻ���û��������A�>�6�5�4�5�A�N�T�X�Y�N�A�A�A�A�A�A�A�A�g�c�d�g�n�s�������������s�i�g�g�g�g�g�g�лû������������ûлܻ�����
���ܻ������������������������������������������������������������ùܹ�������޹ܹϹù��Y�N�Y�Y�e�r�x�r�n�e�Y�Y�Y�Y�Y�Y�Y�Y�Y�Y�-�&�%�-�8�:�=�F�N�F�E�:�-�-�-�-�-�-�-�-���������������	��������	�����h�g�b�h�m�tāčęĚĢĚĒčā�t�h�h�h�h��¿²�}�~¦²�����	��
�����众���������������ʼּ����޼ּʼ������лͻɻлӻܻ����߻ܻлллллллл�����������}�������������ûл޻ۻлû�ŹŸűŹŻ��������������������������ŹŹ�l�`�`�`�k�l�l�v�x�y�x�w�l�l�l�l�l�l�l�l�<�8�/�/�/�/�7�<�H�U�a�f�a�U�Q�K�I�H�<�<�*�$�*�4�6�C�G�O�U�O�C�6�*�*�*�*�*�*�*�*�Z�T�M�H�M�Z�f�s�u�s�f�^�Z�Z�Z�Z�Z�Z�Z�Z�.������ ���!�9�S���������������l�S�.ĿķĮĲĿ����������
�����
������Ŀ�V�N�I�=�9�4�=�I�V�b�o�{�~ǈǈǈ�{�o�b�V�����������
��#�%�&�'�#��
�������������T�J�K�T�`�m�n�m�m�`�T�T�T�T�T�T�T�T�T�TììùþùìììàÙàììììììììì�������������������ľɾ������������������������������Ŀѿӿݿ޿ݿؿѿĿ���������EEEE!E*E7E@ECEPE\E]E\EXEPEEECE7E*EEECEBECECEPE\EcE^E\EPECECECECECECECECECEC < < b X U j ' 7 B < 8 > k L N $ i  ^ 6 X G O U Z z T � O E  o %  W v 0 [ ) P W 6 2 V r 4 H 8 K w (  9 k 7 A 7 U > O 9 J Q 3 j X c Z S R r L , * 9 ? p % r 6    E  V  j  �  V  /  +  �    n  q  �    7     �  p  n  m  K  W  �  �    "  �  �      �  H  !  �  }  p  �    >  �  )  �  $          �  �  �  1  �  w  �  �  I  �  <    <  �  �  G  S  �  	  �  �  S  5  �  ,  �  �  Y    f  k  	  1  BR  BR  BR  BR  BR  BR  BR  BR  BR  BR  BR  BR  BR  BR  BR  BR  BR  BR  BR  BR  BR  BR  BR  BR  BR  BR  BR  BR  BR  BR  BR  BR  BR  BR  BR  BR  BR  BR  BR  BR  BR  BR  BR  BR  BR  BR  BR  BR  BR  BR  BR  BR  BR  BR  BR  BR  BR  BR  BR  BR  BR  BR  BR  BR  BR  BR  BR  BR  BR  BR  BR  BR  BR  BR  BR  BR  BR  BR  BR  BR  �  �  �  �  �  �  �  �  �  |  g  R  >  .      	  �  �  �    	    �  �  �  �  �  �  �  �  �  n  \  R  Q  P  V  _  h  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  b  ^  Q  2    �  �  y  >     �      �  %  
  r  �  �   �  )  >  O  U  [  b  f  h  h  g  d  ^  W    �  �  �  Z  ,  �        �  �  �  �  �  �  �  �  �  {  g  M  4      �  �  �  �  �  �  �  �  �  �  n  P  /  	  �  �  l  '  �  G  �   �    )  0  "    �  �  �  �  X  )  �  �  �  t  D    �  �  �    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  y  t  p  k  f  \  R  H  =  1  %      �  �  �  �  �  �  {  6  -  #          �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �    x  r  l  f  `  Z  R  F  ;  0  $      �  �  �  �  u  V  1    �  �  �  l  H    �  ^    �  z  	  a  �  �    %  ;  G  F  7      �  �  {  H    �  �  z  �    C  M  U  V  W  \  c  g  e  a  [  S  F  2    �  \  �  :  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �    x  q  j  c  �  �  �  �  �  �  �  r  Q  ,    �  �  �  �  P  �  �  �  A  �  �  �  �  �  �  �  �  �  �  �  i  N  3    �  �  �  �  �  1  *  #          �  �  �  �  �  �  �  �  w  \  @  $    {  |  |  |  |  |  }  �  �  �  �  �  �  �  �  �  �  �  �  �    G  Q  L  M  <    �  �  �  H  �  ~    �    r  �    7  '      	  �  �  �  �  �  �  �  �  �  �  k  S  H  @  7  /  a  �  �  �  �  �  �  �  �  �  t  E    �  �  Q  �  �  �   �  �  y  m  a  S  E  3      �  �  �  �  �  k  \  M  @  3  &  �  �  �  �  �  �  �  �  �  �  �  �  �  ~  n  ]  M  <  ,        �  �  �  �  �  w  }  �  \  1    �  �  w  D    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  ~  {  x  u  r  o  l  �  �  �  x  j  Y  p  w  s  j  Z  F  1    �  �  �  �  D  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  ,  /  -  &      �  �  �  �  �  �  s  S  3    �  �  �  _  	  �  �  �  �  �  �  �  �  �  �  �  �  u  i  [  M  ;  (    �  �    +  .  %    �  �  �  �  X  "  �  �    �  �  �  �  �    >  `    �  �  �  �  �  �  Y    �  ]  �  %  ;  1  �  �  �  �  �  �  �  �  �  �  �  �  w  f  T  C  0    
  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  z  i  X      %  )  )  '         �  �  �  �  i  5  �  �  O  �  �    �  �  �  �  �  �  �  �  �  ~  g  A    �  �  �  �  �  �  -  B  F  H  K  N  O  E  7  "    �  �  �  S    �  <  Y  �  c  f  c  \  Q  >  #  �  �  �  �  �  |  B  Y  V  ?    �   �  {  �  �  �  �  s  ^  A    �  �  �  W    �  }  �  �  �  v  �  �  �  �  �  �  �  _  +    �  �  `  -  �  �  `  �  �  4  �  �  �  {  t  j  ^  M  <  )      �  �  �  �  �  �  �  j  �  �  �  �  �  �  �  �  �  �  �  {  s  k  c  ^  K  %  �  �  �  �  �  �  �  �  �  �  �  �  l  T  B  =    �  z  0    �  �  �  �  �  v  O  &  �  �  �  �  Z  1  	  �  �  �  g     �  
  �  �  �  �  �  �  �  �  �  �  p  V  <  "     �   �   �   z  M  M  E  ;  +    	  �  �  �      �  �  �  �  s  7  �  �  �  �  �  �  p  [  ?        �  �  �  Q    �  �  �  I  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  w  d  �  �  �  �  �  �  �  �  �  �  �  {  Z  0    �  v  !  �  0  �    	    �  �  �  �  z  B    �  g    �  (  �    �  �  �  �  �  �  �  �  �  �  �  �  �  {  v  o  f  ]  T  K  B  9            
  �  �  �  �  �  �  |  W  2     �   �   �   �    '  0  4  =  I  S  U  O  D  6  $    �  �  �  I  �  K   �  �  �  �  �  �  �  �  �  �  �  k  :    �  ~  -  �  c  �  �  �    V  �  �  �  �  �  `  9    �  �    �  �  Y  �  �    �  n  X  7    �  �  �  �  �  �  ~  _  ?    �  �  �  r  E  �  �  �  �  �  �  �  �  �  {  i  V  B  ,    �  �  �  �  �  W  V  T  P  H  ?  0      �  �  �  �  �  �  u  T  )  �  �  D  f  �  �  �  �  �  �  �  p  4  �  U  �    i  �  �  /   �  �  �     
  �  �  �  �  U  !  �  �  o    �  <  �  *  �  J  w  d  R  ?  )       �  �  w  8  �  �  +  �  I  �  4  �  �  y  m  a  I  0    �  �  �  �  ]  3  	  �  �  u  ;    �  �    ;  +  &  %    �  �  �  �  �  �  ~  7  �  �  ]    �    �  �  �  �  �  �  �  t  x  n  J    �  �  m  )  �  �  M   �  �  �  ~  q  _  N  7      �  �  �  �  i  N  :  &  0  G  ]  !                      �  �  �  �  �  �  �  {  W  ~  }  |  {  z  y  x  w  v  u  t  u  u  v  v  w  w  x  x  x  �  �  �  �  �  �  �  �  y  j  [  L  =  -       �   �   �   �  B  L  A  /  D  F  0    �  �  �  �  �  m  /  �  �    �  k  ]  d  f  d  Z  F  ,    �  �  �    ?    �  �  v  K    �  �  �  �  y  ]  9    �  �  �  _  #  �  �  E  �  b  �  Z  �  �  �  �  �  �  �  �  �  �  �  �  �  �  u  [  :    �  �  ^  d  ]  W  Q  K  B  2  #      �  �  �  �  �  �  �  �  �  v  I  !  �  �  �  |  Q    �  �  q  4  �  �  y  9  �  �  t  1              �  �  �  �  �  �  �  �  �  �  �  x  ]  A    �  �  �    V  +  �  �  �  [    �  j  �  _  �  7  �    �  l  Z  O  N  P  A  &  �  �  �  �  f  A  �  E    �  M  u  r  c  T  G  =  /      �  �  �  �  n  Q  5    �  �  �  